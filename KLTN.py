import os
import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pymannkendall as mk
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import shap

# 1. CẤU HÌNH HỆ THỐNG & RANH GIỚI PHẠM VI

PROJECT_ID = 'du-lieu-508707'

os.makedirs("data/processed", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

def init_gee(project_id):
    try:
        ee.Initialize(project=project_id)
        print("Đã kết nối Google Earth Engine thành công.")
    except Exception:
        print("Khởi tạo lại xác thực Google Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project=project_id)

init_gee(PROJECT_ID)

# Lấy ranh giới Việt Nam và đơn giản hóa hình học (simplify 1000m) để giải phóng bộ nhớ
ROI_VN = ee.FeatureCollection("FAO/GAUL/2015/level0") \
           .filter(ee.Filter.eq('ADM0_NAME', 'Viet Nam')) \
           .geometry().simplify(maxError=1000)


# 2. TRÍCH XUẤT CHUỖI DỮ LIỆU TỪNG THÁNG (2016 - 2026)

def fetch_vietnam_timeseries(start_year=2016, end_year=2026):
    print(f"[1/4] Đang xử lý dữ liệu đa vệ tinh từ năm {start_year} đến {end_year} (132 tháng)...")
    
    start_date = f"{start_year}-01-01"
    end_date = f"{end_year+1}-01-01"
    
    # 2.1. Chuẩn bị Collections Vệ tinh
    def prep_landsat(img):
        qa = img.select('QA_PIXEL')
        mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        optical = img.select('SR_B.').multiply(0.0000275).add(-0.2)
        return img.addBands(optical, None, True).updateMask(mask)

    def add_indices(img):
        nir, red, blue = img.select('SR_B5'), img.select('SR_B4'), img.select('SR_B2')
        ndvi = img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        evi_denom = nir.add(red.multiply(6.0)).subtract(blue.multiply(7.5)).add(1.0)
        evi = nir.subtract(red).multiply(2.5).divide(evi_denom).rename('EVI')
        savi = nir.subtract(red).multiply(1.5).divide(nir.add(red).add(0.5)).rename('SAVI')
        lai = evi.multiply(3.618).subtract(0.118).rename('LAI')
        return img.addBands([ndvi, evi, savi, lai])

    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(ROI_VN).filterDate(start_date, end_date).map(prep_landsat).map(add_indices)
    l9 = ee.ImageCollection('LANDSAT/LC09/C02/T1_L2').filterBounds(ROI_VN).filterDate(start_date, end_date).map(prep_landsat).map(add_indices)
    landsat_col = l8.merge(l9)

    s5p_no2 = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2").filterBounds(ROI_VN).filterDate(start_date, end_date).select('tropospheric_NO2_column_number_density')
    modis_aod = ee.ImageCollection("MODIS/061/MCD19A2_GRANULES").filterBounds(ROI_VN).filterDate(start_date, end_date).select('Optical_Depth_047')
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR").filterBounds(ROI_VN).filterDate(start_date, end_date)

    # Tạo danh sách các tháng từ 2016-01 đến 2026-12
    dates_list = [f"{y}-{m:02d}-01" for y in range(start_year, end_year + 1) for m in range(1, 13)]

    results = []
    # Vòng lặp giải phóng bộ nhớ từng tháng
    for i, d_str in enumerate(dates_list, 1):
        print(f"  -> [{i}/{len(dates_list)}] Đang trích xuất tháng: {d_str}...", end="\r")
        s = ee.Date(d_str)
        e = s.advance(1, 'month')
        
        # Landsat
        f_ls = landsat_col.filterDate(s, e)
        s_ls = ee.Dictionary(ee.Algorithms.If(
            f_ls.size().gt(0),
            f_ls.select(['NDVI', 'EVI', 'SAVI', 'LAI']).median().reduceRegion(
                reducer=ee.Reducer.mean(), geometry=ROI_VN, scale=3000, maxPixels=1e13
            ),
            {'NDVI': None, 'EVI': None, 'SAVI': None, 'LAI': None}
        ))
        
        # Sentinel-5P NO2 (Tự động trả về None nếu trước T7/2018)
        f_s5p = s5p_no2.filterDate(s, e)
        s_s5p = ee.Dictionary(ee.Algorithms.If(
            f_s5p.size().gt(0),
            f_s5p.mean().reduceRegion(
                reducer=ee.Reducer.mean(), geometry=ROI_VN, scale=5000, maxPixels=1e13
            ),
            {'tropospheric_NO2_column_number_density': None}
        ))

        # MODIS AOD
        f_aod = modis_aod.filterDate(s, e)
        s_aod = ee.Dictionary(ee.Algorithms.If(
            f_aod.size().gt(0),
            f_aod.mean().multiply(0.001).reduceRegion(
                reducer=ee.Reducer.mean(), geometry=ROI_VN, scale=5000, maxPixels=1e13
            ),
            {'Optical_Depth_047': None}
        ))

        # ERA5-Land
        f_era = era5.filterDate(s, e)
        s_era = ee.Dictionary(ee.Algorithms.If(
            f_era.size().gt(0),
            f_era.mean().select(['temperature_2m', 'total_precipitation_sum', 'u_component_of_wind_10m', 'v_component_of_wind_10m'])
                 .reduceRegion(reducer=ee.Reducer.mean(), geometry=ROI_VN, scale=10000, maxPixels=1e13),
            {'temperature_2m': None, 'total_precipitation_sum': None, 'u_component_of_wind_10m': None, 'v_component_of_wind_10m': None}
        ))

        merged = s_ls.combine(s_s5p).combine(s_aod).combine(s_era)
        feat = ee.Feature(None, merged).set('date', d_str)
        
        res_dict = feat.getInfo()['properties']
        results.append(res_dict)

    print("\n Đã hoàn thành tải toàn bộ chuỗi thời gian 2016-2026!")
    df = pd.DataFrame(results)
    
    df['date'] = pd.to_datetime(df['date'])
    df.rename(columns={
        'tropospheric_NO2_column_number_density': 'NO2_S5P',
        'Optical_Depth_047': 'AOD_PM25_Proxy',
        'temperature_2m': 'temp_ERA5',
        'total_precipitation_sum': 'precip_ERA5'
    }, inplace=True)
    
    if 'u_component_of_wind_10m' in df.columns and 'v_component_of_wind_10m' in df.columns:
        df['wind_ERA5'] = np.sqrt(df['u_component_of_wind_10m']**2 + df['v_component_of_wind_10m']**2)
        df.drop(columns=['u_component_of_wind_10m', 'v_component_of_wind_10m'], inplace=True, errors='ignore')
        
    if 'temp_ERA5' in df.columns:
        df['temp_ERA5'] = df['temp_ERA5'] - 273.15

    return df.sort_values('date').reset_index(drop=True)


# 3. TIỀN XỬ LÝ & NỘI SUY DỮ LIỆU KHUYẾT TẤT (PCHIP)

def clean_and_preprocess(df):
    print("[2/4] Đang tiền xử lý, nội suy làm mịn nhiễu và xử lý dữ liệu khuyết...")
    df.set_index('date', inplace=True)
    
    cols = ['NDVI', 'EVI', 'SAVI', 'LAI', 'NO2_S5P', 'AOD_PM25_Proxy', 'temp_ERA5', 'precip_ERA5', 'wind_ERA5']
    for col in cols:
        if col in df.columns:
            # PCHIP nội suy mượt các điểm NaN (bao gồm NO2 giai đoạn 2016 - T6/2018)
            df[col] = df[col].interpolate(method='pchip').bfill().ffill()

    # Làm mịn Savitzky-Golay với window_length=7 cho chuỗi thời gian dài 132 tháng
    df['NDVI_smooth'] = savgol_filter(df['NDVI'], window_length=7, polyorder=2)
    df['EVI_smooth'] = savgol_filter(df['EVI'], window_length=7, polyorder=2)
    
    return df.dropna()


# 4. CHẨN ĐOÁN THỐNG KÊ & MÔ HÌNH HỌC MÁY (10 NĂM)

def run_analysis(df):
    print("\n" + "="*65)
    print(" 1. KIỂM ĐỊNH XU HƯỚNG MANN-KENDALL MẢNG XANH (2016 - 2026)")
    print("="*65)
    mk_res = mk.original_test(df['NDVI_smooth'])
    print(f"• Xu hướng diễn biến: {mk_res.trend}")
    print(f"• Mức ý nghĩa p-value: {mk_res.p:.4f}")
    print(f"• Hệ số góc Sen (Sen's Slope): {mk_res.slope:.6f} / tháng")

    stl_ndvi = STL(df['NDVI_smooth'], period=12).fit()
    stl_aod = STL(df['AOD_PM25_Proxy'], period=12).fit()
    
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.set_xlabel('Thời gian (2016 - 2026)')
    ax1.set_ylabel('Xu hướng NDVI (Đã tách mùa)', color='tab:green')
    ax1.plot(df.index, stl_ndvi.trend, color='tab:green', linewidth=2.5, label='NDVI Trend')
    ax1.tick_params(axis='y', labelcolor='tab:green')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Xu hướng AOD / PM2.5 (MODIS)', color='tab:red')
    ax2.plot(df.index, stl_aod.trend, color='tab:red', linewidth=2, linestyle='--', label='AOD Trend')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title("Phân rã STL 10 năm: Tương quan Xu hướng Mảng xanh và Ô nhiễm tại Việt Nam")
    fig.tight_layout()
    plt.savefig("outputs/figures/stl_decomposition_vietnam_10y.png", dpi=300)
    plt.close()
    print("Đã xuất biểu đồ STL 10 năm: outputs/figures/stl_decomposition_vietnam_10y.png")

    features = ['NDVI_smooth', 'EVI_smooth', 'SAVI', 'LAI', 'NO2_S5P', 'temp_ERA5', 'precip_ERA5', 'wind_ERA5']
    X = df[features]
    y = df['AOD_PM25_Proxy']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    y_pred = model.predict(X)
    print("\n" + "="*65)
    print(" 2. ĐÁNH GIÁ MÔ HÌNH RANDOM FOREST TOÀN QUỐC (2016 - 2026)")
    print("="*65)
    print(f"• Hệ số xác định R²: {r2_score(y, y_pred):.3f}")
    print(f"• Sai số RMSE: {np.sqrt(mean_squared_error(y, y_pred)):.4f}")

    sample_base = X.mean().to_frame().T
    sample_inc = sample_base.copy()
    sample_inc['NDVI_smooth'] += 0.01

    delta_aod = model.predict(sample_inc)[0] - model.predict(sample_base)[0]
    print("\n" + "="*65)
    print(" 3. KẾT QUẢ ĐỊNH LƯỢNG ĐỘ NHẠY TÁC ĐỘNG")
    print("="*65)
    print(f"Khi chỉ số phủ xanh NDVI toàn quốc tăng 1% (+0.01), chỉ số ô nhiễm AOD thay đổi: {delta_aod:.6f}")

    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    plt.title("Tầm quan trọng của các yếu tố đến ô nhiễm không khí tại Việt Nam (2016-2026)")
    plt.tight_layout()
    plt.savefig("outputs/figures/shap_bar_plot_vietnam_10y.png", dpi=300)
    plt.close()
    print("Đã xuất biểu đồ SHAP 10 năm: outputs/figures/shap_bar_plot_vietnam_10y.png")


# 5. XUẤT BẢN ĐỒ RASTER NĂM MỚI NHẤT 2026 (GEE DRIVE EXPORT)

def export_ndvi_map_vietnam(year=2026):
    print(f"\n[4/4] Khởi tạo tiến trình xuất bản đồ NDVI 100m năm {year} về Google Drive...")
    start_date = f"{year}-01-01"
    end_date = f"{year}-12-31"
    
    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2').filterBounds(ROI_VN).filterDate(start_date, end_date)
    
    def get_ndvi(img):
        return img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')
        
    ndvi_annual = l8.map(get_ndvi).median().clip(ROI_VN)
    
    task = ee.batch.Export.image.toDrive(
        image=ndvi_annual,
        description=f'Ban_Do_NDVI_VietNam_{year}',
        folder='GEE_Exports',
        fileNamePrefix=f'NDVI_VietNam_{year}_100m',
        region=ROI_VN,
        scale=100,
        maxPixels=1e13
    )
    task.start()
    print(f"Đã gửi lệnh xuất bản đồ lên GEE. Kiểm tra thư mục 'GEE_Exports' trên Google Drive.")


# MAIN PIPELINE
if __name__ == "__main__":
    df_raw = fetch_vietnam_timeseries(start_year=2016, end_year=2026)
    
    df_clean = clean_and_preprocess(df_raw)
    
    output_csv = "data/processed/du_lieu_khoa_luan_vietnam_2016_2026.csv"
    df_clean.to_csv(output_csv)
    print(f"Đã xuất file CSV dữ liệu 10 năm tại: {output_csv}")

    run_analysis(df_clean)
    
    export_ndvi_map_vietnam(year=2026)