import json

def test_sorting():
    data = json.load(open('structuredData.json'))
    elements = [e for e in data['elements'] if 'Text' in e and e['Text'].strip()]
    
    def sort_key(elem):
        b = elem['Bounds']
        path = elem['Path']
        
        # Parse path to get structural order
        path_parts = path.split('/')[2:]  # Remove empty first two parts
        
        # Create a sortable path key (all as strings)
        path_key = []
        for part in path_parts:
            if '[' in part:
                # Handle indexed parts like P[2]
                base, index = part.split('[')
                index = index.rstrip(']')
                path_key.extend([base, index if index.isdigit() else '0'])
            else:
                # Handle unindexed parts - treat as index 1
                path_key.extend([part, '1'])
        
        if len(b) >= 4:
            # Round Y to nearest 5 points to group similar Y positions
            y_group = round(b[1] / 5) * 5
            return (elem['Page'], path_key, -y_group, b[0])
        return (elem['Page'], path_key, 0, 0)
    
    elements.sort(key=sort_key)
    print('NEW SORTING WITH PATH:')
    for i, e in enumerate(elements[:15]):
        print(f'{i+1:2d}. {e["Path"]} Y={e["Bounds"][1]:.1f}, X={e["Bounds"][0]:.1f}: "{e["Text"]}"')

if __name__ == "__main__":
    test_sorting() 