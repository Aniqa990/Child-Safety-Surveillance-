# import os
# from pathlib import Path

# # Change this if your folders are located somewhere else
# SPLIT_ROOT = "data" 

# def check_data_leakage():
#     splits = ["train", "val", "test"]
#     video_sets = {"train": set(), "val": set(), "test": set()}

#     print("🔍 Scanning folders for Data Leakage...\n")

#     # 1. Gather the true base names of all videos in each folder
#     for split in splits:
#         split_dir = Path(SPLIT_ROOT) / split
#         if not split_dir.exists():
#             continue
            
#         for npy_file in split_dir.rglob("*.npy"):
#             name = npy_file.stem
            
#             # Strip the augmentation text to find the true source name
#             if "Copy of " in name:
#                 name = name.split("Copy of ")[-1]
                
#             # Strip the clip number (e.g., "fi001_clip000" becomes "fi001")
#             base_name = name.split("_clip")[0]
            
#             video_sets[split].add(base_name)

#     # 2. Check for overlaps (Leakage)
#     train_val_leak = video_sets["train"].intersection(video_sets["val"])
#     train_test_leak = video_sets["train"].intersection(video_sets["test"])
#     val_test_leak = video_sets["val"].intersection(video_sets["test"])

#     # 3. Print the results
#     if not train_val_leak and not train_test_leak and not val_test_leak:
#         print("✅ SUCCESS: Your data is perfectly split! No source videos overlap.")
#     else:
#         print("🚨 WARNING: DATA LEAKAGE DETECTED 🚨")
#         if train_val_leak:
#             print(f"\n❌ These {len(train_val_leak)} videos are in BOTH Train and Val:")
#             print(list(train_val_leak)[:10]) # Prints up to 10 examples
            
#         if train_test_leak:
#             print(f"\n❌ These {len(train_test_leak)} videos are in BOTH Train and Test:")
#             print(list(train_test_leak)[:10])
            
#         if val_test_leak:
#             print(f"\n❌ These {len(val_test_leak)} videos are in BOTH Val and Test:")
#             print(list(val_test_leak)[:10])

# if __name__ == "__main__":
#     check_data_leakage()

import os
from pathlib import Path
from collections import defaultdict

# Change this if your folders are located somewhere else
SPLIT_ROOT = "data" 

def check_data_leakage_detailed():
    splits = ["train", "val", "test"]
    
    # This will store: video_files["train"]["climbing_042"] = ["climbing_042_clip001.npy", "aug_Brightness..."]
    video_files = {
        "train": defaultdict(list),
        "val": defaultdict(list),
        "test": defaultdict(list)
    }

    print("🔍 Scanning folders for Data Leakage...\n")

    # 1. Gather files and group them by their true base names
    for split in splits:
        split_dir = Path(SPLIT_ROOT) / split
        if not split_dir.exists():
            continue
            
        for npy_file in split_dir.rglob("*.npy"):
            name = npy_file.stem
            
            # Strip the augmentation text
            if "Copy of " in name:
                name = name.split("Copy of ")[-1]
                
            # Strip the clip number to find the base name
            base_name = name.split("_clip")[0]
            
            # Save the FULL file name under its base group
            video_files[split][base_name].append(npy_file.name)

    # 2. Check for overlaps by looking at the keys (base names)
    train_val_leak = set(video_files["train"].keys()).intersection(set(video_files["val"].keys()))
    train_test_leak = set(video_files["train"].keys()).intersection(set(video_files["test"].keys()))
    val_test_leak = set(video_files["val"].keys()).intersection(set(video_files["test"].keys()))

    # 3. Print the detailed results
    if not train_val_leak and not train_test_leak and not val_test_leak:
        print("✅ SUCCESS: Your data is perfectly split! No source videos overlap.")
        return

    print("🚨 WARNING: DATA LEAKAGE DETECTED 🚨\n")
    
    def print_leakage_details(leak_set, split1, split2):
        if not leak_set:
            return
        print(f"❌ {len(leak_set)} source videos overlap between {split1.upper()} and {split2.upper()}:")
        
        # We'll print the full details for the first 5 leaking videos to avoid flooding the terminal
        for base_name in list(leak_set)[:5]:
            print(f"\n   📹 Video Group: '{base_name}'")
            print(f"   -> Found in {split1.upper()}:")
            for f in video_files[split1][base_name]:
                print(f"      - {f}")
                
            print(f"   -> Found in {split2.upper()}:")
            for f in video_files[split2][base_name]:
                print(f"      - {f}")
                
        if len(leak_set) > 5:
            print(f"\n   ... plus {len(leak_set) - 5} more leaking video groups (hidden to save space).")
        print("-" * 60)

    print_leakage_details(train_val_leak, "train", "val")
    print_leakage_details(train_test_leak, "train", "test")
    print_leakage_details(val_test_leak, "val", "test")

if __name__ == "__main__":
    check_data_leakage_detailed()