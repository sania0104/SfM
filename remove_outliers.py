import numpy as np

def clean_ply(input_file, output_file):
    print(f"Reading {input_file}...")
    
    with open(input_file, 'r') as f:
        lines = f.readlines()
    
    header = []
    points = []
    
    header_end = 0
    for i, line in enumerate(lines):
        if "end_header" in line:
            header.append(line)
            header_end = i + 1
            break
        header.append(line)
        
    # Parse data lines
    data_lines = lines[header_end:]
    
    valid_lines = []
    
    # Filter loop
    for line in data_lines:
        parts = line.split()
        if len(parts) < 3: continue
        
        try:
            x = float(parts[0])
            y = float(parts[1])
            z = float(parts[2])
            
            # keep anything between -10 and 10.
            if abs(x) < 10.0 and abs(y) < 10.0 and abs(z) < 10.0:
                valid_lines.append(line)
                
        except ValueError:
            continue

    print(f"Original points: {len(data_lines)}")
    print(f"Cleaned points:  {len(valid_lines)}")
    print(f"Removed: {len(data_lines) - len(valid_lines)} outliers")
    
    new_header = []
    for h in header:
        if "element vertex" in h:
            new_header.append(f"element vertex {len(valid_lines)}\n")
        else:
            new_header.append(h)
            
    with open(output_file, 'w') as f:
        f.writelines(new_header)
        f.writelines(valid_lines)
        
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    clean_ply("reconstruction_result.ply", "reconstruction_clean.ply")