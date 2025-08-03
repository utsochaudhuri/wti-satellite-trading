import os
import numpy as np
import pandas as pd
import time
import datetime
from sentinelhub import (
    SHConfig, BBox, CRS,
    SentinelHubRequest, DataCollection, MimeType
)

# ====== CONFIGURE CREDENTIALS ======
config = SHConfig()
config.sh_client_id = '285a8f71-e9f6-4a17-9deb-e19b2f6352a4'
config.sh_client_secret = 'yeGAtfOYgWH1jj2UEVFwhfubWqJNbUNe'

# Test credentials immediately
print("Testing SentinelHub credentials...")
try:
    from sentinelhub import SentinelHubCatalog
    catalog = SentinelHubCatalog(config=config)
    print("✅ Credentials verified successfully")
except Exception as e:
    print(f" Credential error: {e}")
    exit(1)

# ====== DEFINE CITY BOUNDING BOXES ======
cities = {
    "Tulsa": BBox(bbox=[-96.227, 35.871, -95.652, 36.308], crs=CRS.WGS84),
    "Oklahoma_City": BBox(bbox=[-97.759, 35.287, -97.237, 35.715], crs=CRS.WGS84)
}

print("Cities to analyze:")
for city, bbox in cities.items():
    print(f"   {city}: {bbox}")

# ====== JULY 2024 DAILY DATES ======
def get_date_list(start_date, end_date):
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current.isoformat())
        current += datetime.timedelta(days=1)
    return dates

start = datetime.date(2018, 1, 1)
end = datetime.date.today()  # 2025-08-01 in your environment
dates = get_date_list(start, end)[:-1]

# ====== OUTPUT FOLDER ======
output_folder = "oklahoma_cloud_analysis"
os.makedirs(output_folder, exist_ok=True)
print(f"Output folder: {output_folder}")

# ====== CLOUD ANALYSIS EVALSCRIPT (NO IMAGE DOWNLOAD) ======
evalscript = """
//VERSION=3
function setup() {
    return {
        input: ["B04", "B03", "B02", "CLM"],
        output: { bands: 4, sampleType: "UINT8" }
    };
}

function evaluatePixel(sample) {
    // Return RGB values and cloud mask
    return [
        Math.min(255, Math.max(0, sample.B04 * 255 * 2.5)),
        Math.min(255, Math.max(0, sample.B03 * 255 * 2.5)), 
        Math.min(255, Math.max(0, sample.B02 * 255 * 2.5)),
        sample.CLM * 255
    ];
}
"""

# ====== CLOUD COVERAGE ANALYSIS FUNCTION ======
def analyze_cloud_coverage(date_str, city_name, bbox, timeout=20):
    """Get cloud coverage without downloading image files"""
    start_time = time.time()
    print(f"{city_name} {date_str}...", end="", flush=True)
    
    try:
        request = SentinelHubRequest(
            evalscript=evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L2A,
                    time_interval=(date_str, date_str),
                    mosaicking_order='leastCC',
                    maxcc=1.0
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=bbox,
            size=(128, 128),  # Small size just for analysis
            config=config
        )
        
        data = request.get_data()
        elapsed = time.time() - start_time
        
        if data and len(data) > 0 and data[0] is not None:
            arr = np.array(data[0])
            if arr.size > 0:
                # Check if image is completely black/empty
                rgb_data = arr[:, :, :3]
                avg_brightness = np.mean(rgb_data)
                max_brightness = np.max(rgb_data)
                
                # Skip completely black images
                if avg_brightness < 5 and max_brightness < 20:
                    print(f" SKIPPED (black) ({elapsed:.1f}s)")
                    return None, elapsed
                
                # Calculate cloud coverage from 4th channel (CLM)
                if arr.shape[-1] >= 4:
                    cloud_mask = arr[:, :, 3]
                    cloud_coverage = (np.sum(cloud_mask > 127) / cloud_mask.size) * 100
                    cloud_coverage = round(cloud_coverage, 1)
                else:
                    cloud_coverage = None
                
                print(f" {cloud_coverage}% clouds ({elapsed:.1f}s)")
                return cloud_coverage, elapsed
            else:
                print(f" Empty ({elapsed:.1f}s)")
                return None, elapsed
        else:
            print(f" No data ({elapsed:.1f}s)")
            return None, elapsed
            
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        if "timeout" in error_msg.lower():
            print(f" TIMEOUT ({elapsed:.1f}s)")
        elif "rate limit" in error_msg.lower():
            print(f" RATE LIMITED ({elapsed:.1f}s)")
        else:
            print(f" ERROR ({elapsed:.1f}s)")
        return None, elapsed

# ====== MAIN EXECUTION ======
def main():
    print("\n" + "="*70)
    print("OKLAHOMA CLOUD COVERAGE ANALYZER")
    print("="*70)
    
    cloud_data = []
    total_requests = len(dates) * len(cities)
    successful_requests = 0
    total_time = 0
    
    print(f"Starting analysis of {total_requests} city-date combinations...")
    overall_start = time.time()
    
    request_count = 0
    
    for date_idx, date in enumerate(dates, 1):
        print(f"\n📅 [{date_idx:2d}/{len(dates)}] {date}")
        
        for city_name, bbox in cities.items():
            request_count += 1
            progress = f"[{request_count:3d}/{total_requests}]"
            print(f"  {progress} ", end="")
            
            # Analyze cloud coverage
            cloud_pct, elapsed = analyze_cloud_coverage(date, city_name, bbox)
            total_time += elapsed
            
            if cloud_pct is not None:
                successful_requests += 1
                cloud_data.append({
                    'date': date,
                    'city': city_name,
                    'cloud_coverage_percent': cloud_pct
                })
            
            # Brief pause between requests
            time.sleep(0.3)
        
        # Show progress every 5 dates
        if date_idx % 5 == 0:
            avg_time = total_time / request_count if request_count > 0 else 0
            success_rate = (successful_requests / request_count) * 100 if request_count > 0 else 0
            print(f"Progress: {successful_requests}/{request_count} successful ({success_rate:.1f}%), avg {avg_time:.1f}s/request")
    
    # Save results to CSV
    if cloud_data:
        csv_file = os.path.join(output_folder, "oklahoma_cloud_coverage_July2024.csv")
        df = pd.DataFrame(cloud_data)
        aggregated = []
        for date, group in df.groupby('date'):
            if len(group) == 2:
                avg_cloud = round(group['cloud_coverage_percent'].mean(), 1)
                aggregated.append({'date': date, 'city': 'averaged', 'cloud_coverage_percent': avg_cloud})
            else:
                aggregated.append(group.iloc[0].to_dict())

        df = pd.DataFrame(aggregated)
        df.to_csv(csv_file, index=False)
    else:
        print("No valid cloud data collected")

if __name__ == "__main__":
    main()