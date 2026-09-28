import os
import ee
import folium

# 1. CẤU HÌNH HỆ THỐNG & TRÍCH XUẤT RANH GIỚI VIỆT NAM

PROJECT_ID = 'du-lieu-508707'
OUTPUT_DIR = "outputs/maps"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def init_gee(project_id):
    try:
        ee.Initialize(project=project_id)
        print("Đã kết nối thành công Google Earth Engine.")
    except Exception:
        print("Xác thực lại Google Earth Engine...")
        ee.Authenticate()
        ee.Initialize(project=project_id)

init_gee(PROJECT_ID)

vietnam_fc = ee.FeatureCollection("FAO/GAUL/2015/level0").filter(ee.Filter.eq('ADM0_NAME', 'Viet Nam'))
vietnam_geom = vietnam_fc.geometry().simplify(maxError=500)
vietnam_geojson = vietnam_fc.getInfo()

ESRI_LIGHT_GRAY_URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'


# 2. MAP 1: BẢN ĐỒ ĐỘ PHỦ XANH & PHÂN CẤP LỚP PHỦ ĐẤT

def create_greenery_map():
    print("\n[1/2] Đang tạo Map 1: Bản đồ Phân cấp Độ phủ xanh & Lớp phủ...")

    # 2.1. Tính chỉ số NDVI từ Landsat 8/9
    def get_ndvi(image):
        qa = image.select('QA_PIXEL')
        mask = qa.bitwiseAnd(1 << 3).eq(0).And(qa.bitwiseAnd(1 << 4).eq(0))
        optical = image.select('SR_B.').multiply(0.0000275).add(-0.2)
        img = image.addBands(optical, None, True).updateMask(mask)
        return img.normalizedDifference(['SR_B5', 'SR_B4']).rename('NDVI')

    l8 = ee.ImageCollection('LANDSAT/LC08/C02/T1_L2') \
        .filterBounds(vietnam_geom) \
        .filterDate('2024-01-01', '2026-01-01') \
        .map(get_ndvi)
    ndvi = l8.median().clip(vietnam_geom)

    # 2.2. Kết hợp ESA WorldCover v200 và phân cấp mật độ rừng theo NDVI
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select('Map').clip(vietnam_geom)
    is_tree = worldcover.eq(10)

    from_list = [80, 90, 95, 40, 50, 60, 20, 30]
    to_list   = [ 1,  4,  4,  5,  6,  7,  8,  8]
    lc_other = worldcover.remap(from_list, to_list)

    # Phân cấp: 2 (Rừng rậm: NDVI >= 0.70), 3 (Rừng thưa: NDVI < 0.70)
    lc_remapped = ee.Image(0) \
        .where(lc_other.eq(1), 1) \
        .where(is_tree.And(ndvi.gte(0.70)), 2) \
        .where(is_tree.And(ndvi.lt(0.70)), 3) \
        .where(lc_other.eq(4), 4) \
        .where(lc_other.eq(5), 5) \
        .where(lc_other.eq(6), 6) \
        .where(lc_other.eq(7), 7) \
        .where(lc_other.eq(8), 8) \
        .updateMask(worldcover.gt(0))

    # 2.3. Dựng bản đồ Folium
    m1 = folium.Map(location=[16.0, 108.0], zoom_start=6, tiles="OpenStreetMap")
    folium.TileLayer(tiles=ESRI_LIGHT_GRAY_URL, attr='Esri', name='Bản đồ nền Xám (Esri Light Gray)', overlay=False).add_to(m1)

    legend_colors = [
        '#00e6c3',  # 1: Mặt nước
        '#00451b',  # 2: Rừng rậm (NDVI >= 0.70)
        '#4caf50',  # 3: Rừng thưa (NDVI < 0.70)
        '#3b3ebf',  # 4: Ngập nước
        '#ffff00',  # 5: Trồng trọt
        '#ff0000',  # 6: Dân cư
        '#ffffff',  # 7: Đất trống
        '#c8b185'   # 8: Đồng cỏ/Cây bụi
    ]

    map_id_lc = lc_remapped.getMapId({'min': 1, 'max': 8, 'palette': legend_colors})
    folium.TileLayer(
        tiles=map_id_lc['tile_fetcher'].url_format,
        attr='ESA WorldCover + Landsat NDVI',
        name='Bản đồ Phân cấp Lớp phủ & Mật độ Rừng',
        overlay=True, opacity=0.9
    ).add_to(m1)

    folium.GeoJson(
        vietnam_geojson,
        name="Ranh giới Việt Nam",
        style_function=lambda x: {'color': '#333333', 'weight': 1.0, 'fillOpacity': 0.0}
    ).add_to(m1)

    # Chú giải Lớp phủ
    legend_html1 = '''
    <div style="position: fixed; bottom: 30px; left: 30px; width: 230px; background: white; border: 2px solid grey; z-index:9999; font-size: 13px; font-family: Arial, sans-serif; padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3); border-radius: 4px;">
        <b style="font-size: 14px;">Chú giải Lớp phủ</b><br><hr style="margin: 5px 0;">
        <div style="display:flex; align-items:center; margin-bottom: 5px;"><span style="background:#00e6c3; width:22px; height:14px; border:1px solid #666; margin-right:8px; display:inline-block;"></span>Đất mặt nước</div>
        <div style="display:flex; align-items:center; margin-bottom: 5px;"><span style="background:#00451b; width:22px; height:14px; border:1px solid #666; margin-right:8px; display:inline-block;"></span>Rừng rậm (NDVI &ge; 0.70)</div>
        <div style="display:flex; align-items:center; margin-bottom: 5px;"><span style="background:#4caf50; width:22px; height:14px; border:1px solid #666; margin-right:8px; display:inline-block;"></span>Rừng thưa (NDVI &lt; 0.70)</div>
        <div style="display:flex; align-items:center; margin-bottom: 5px;"><span style="background:#3b3ebf; width:22px; height:14px; border:1px solid #666; margin-right:8px; display:inline-block;"></span>Thực vật ngập nước</div>
        <div style="display:flex; align-items:center; margin-bottom: 5px;"><span style="background:#ffff00; width:22px; height:14px; border:1px solid #666; margin-right:8px; display:inline-block;"></span>Đất trồng trọt</div>
        <div style="display:flex; align-items:center; margin-bottom: 5px;"><span style="background:#ff0000; width:22px; height:14px; border:1px solid #666; margin-right:8px; display:inline-block;"></span>Đất dân cư</div>
        <div style="display:flex; align-items:center; margin-bottom: 5px;"><span style="background:#ffffff; width:22px; height:14px; border:1px solid #666; margin-right:8px; display:inline-block;"></span>Đất trống</div>
        <div style="display:flex; align-items:center; margin-bottom: 5px;"><span style="background:#c8b185; width:22px; height:14px; border:1px solid #666; margin-right:8px; display:inline-block;"></span>Đất đồng cỏ / Cây bụi</div>
    </div>
    '''
    m1.get_root().html.add_child(folium.Element(legend_html1))
    folium.LayerControl(position='topright', collapsed=False).add_to(m1)

    out_path1 = os.path.join(OUTPUT_DIR, "bando_phu_xanh_vietnam.html")
    m1.save(out_path1)
    print(f"Xuất thành công Map 1: {out_path1}")


# 3. MAP 2: BẢN ĐỒ Ô NHIỄM KHÔNG KHÍ (PM2.5 & NO2)

def create_pollution_map():
    print("\n[2/2] Đang tạo Map 2: Bản đồ Ô nhiễm Không khí (PM2.5 & NO2)...")

    # 3.1. Trích xuất PM2.5 (µg/m³) & NO2 (mol/m²)
    pm25_img = ee.ImageCollection("ECMWF/CAMS/NRT") \
        .filterBounds(vietnam_geom) \
        .select('particulate_matter_d_less_than_25_um_surface') \
        .sort('system:time_start', False) \
        .limit(30) \
        .mean() \
        .multiply(1e9) \
        .clip(vietnam_geom)

    no2_img = ee.ImageCollection("COPERNICUS/S5P/OFFL/L3_NO2") \
        .filterBounds(vietnam_geom) \
        .select('tropospheric_NO2_column_number_density') \
        .sort('system:time_start', False) \
        .limit(30) \
        .mean() \
        .clip(vietnam_geom)

    # 3.2. Dựng bản đồ Folium
    m2 = folium.Map(location=[16.0, 108.0], zoom_start=6, tiles="OpenStreetMap")
    folium.TileLayer(tiles=ESRI_LIGHT_GRAY_URL, attr='Esri', name='Bản đồ nền Xám (Esri Light Gray)', overlay=False).add_to(m2)

    palette = ['#2c7bb6', '#ffffbf', '#fdae61', '#d7191c']

    # Lớp Bụi mịn PM2.5
    pm25_mapid = pm25_img.getMapId({'min': 5.0, 'max': 45.0, 'palette': palette})
    folium.TileLayer(
        tiles=pm25_mapid['tile_fetcher'].url_format,
        attr='ECMWF CAMS PM2.5',
        name='Lớp Bụi mịn PM2.5 (µg/m³)',
        overlay=True, opacity=0.8, show=True
    ).add_to(m2)

    # Lớp Khí NO2
    no2_mapid = no2_img.getMapId({'min': 0.00002, 'max': 0.00012, 'palette': palette})
    folium.TileLayer(
        tiles=no2_mapid['tile_fetcher'].url_format,
        attr='Sentinel-5P NO2',
        name='Lớp Khí NO2 (mol/m²)',
        overlay=True, opacity=0.8, show=False
    ).add_to(m2)

    # Nét viền đỏ định hình ranh giới quốc gia
    folium.GeoJson(
        vietnam_geojson,
        name="Ranh giới Việt Nam",
        style_function=lambda x: {'color': "#CB2A2A", 'weight': 1.2, 'fillOpacity': 0.0}
    ).add_to(m2)

    # Chú giải Ô nhiễm
    legend_html2 = '''
    <div style="position: fixed; bottom: 30px; left: 30px; width: 240px; background: white; border: 2px solid #888; z-index:9999; font-size: 12px; font-family: Arial, sans-serif; padding: 10px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3); border-radius: 5px;">
        <b style="font-size: 13px;">Thang Mức Độ Ô Nhiễm</b><br>
        <div style="margin-top: 6px; background: linear-gradient(to right, #2c7bb6, #ffffbf, #fdae61, #d7191c); height: 14px; width: 100%; border-radius: 2px;"></div>
        <div style="display: flex; justify-content: space-between; margin-top: 3px; font-size: 11px; color: #333;">
            <span>Thấp</span><span>Trung bình</span><span>Cao</span>
        </div>
        <hr style="margin: 6px 0;">
        <small>
            • <b>PM2.5:</b> 5.0 – 45.0 µg/m³<br>
            • <b>NO2:</b> 0.00002 – 0.00012 mol/m²
        </small>
    </div>
    '''
    m2.get_root().html.add_child(folium.Element(legend_html2))
    folium.LayerControl(position='topright', collapsed=False).add_to(m2)

    out_path2 = os.path.join(OUTPUT_DIR, "bando_o_nhiem_vietnam.html")
    m2.save(out_path2)
    print(f"✓ Xuất thành công Map 2: {out_path2}")


if __name__ == "__main__":
    create_greenery_map()
    create_pollution_map()
    print("\n Hoàn thành khởi tạo thành công cả 2 bản đồ tương tác!")