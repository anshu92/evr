
import os
import json
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from stock_screener.data.collective2 import Collective2Client, _response_list
from stock_screener.pipeline.collective2_copy import Collective2CopyConfig, _system_id_from_row

def diagnostic():
    cfg = Collective2CopyConfig.from_env()
    print(f"API Key: {cfg.api_key[:4]}...{cfg.api_key[-4:] if len(cfg.api_key) > 8 else ''}")
    
    client = Collective2Client(api_key=cfg.api_key)
    
    print("\nFetching listAllSystems (RAW)...")
    access_rows = []
    try:
        raw_access = client.post("listAllSystems", {})
        print(json.dumps(raw_access, indent=2))
        access_rows = _response_list(raw_access)
        print(f"Parsed access_rows count: {len(access_rows)}")
    except Exception as e:
        print(f"Error in listAllSystems: {e}")

    print("\nFetching getSystemRoster (RAW snippet)...")
    roster_raw = []
    try:
        raw_roster = client.post("getSystemRoster", {"filter": "active"})
        roster_raw = _response_list(raw_roster)
        print(f"Total in roster: {len(roster_raw)}")
        if roster_raw:
             print("Sample of first system raw:")
             print(json.dumps(roster_raw[0], indent=2))
    except Exception as e:
        print(f"Error in getSystemRoster: {e}")

    if roster_raw:
        free_systems = [s for s in roster_raw if str(s.get("monthlyFee")) == "0"]
        print(f"\nFound {len(free_systems)} free systems in roster.")
        for s in free_systems[:5]:
            s_id = s.get("system_id")
            print(f"Testing free system {s_id} ({s.get('system_name')})...")
            try:
                raw_trades = client.post("requestTradesOpen", {"systemid": str(s_id)})
                if raw_trades.get("ok") == "1":
                    print(f"SUCCESS! Accessible free system: {s_id}")
                    break
                else:
                    print(f"Failed: {raw_trades.get('error', {}).get('message')}")
            except Exception as e:
                print(f"Error: {e}")

if __name__ == "__main__":
    diagnostic()
