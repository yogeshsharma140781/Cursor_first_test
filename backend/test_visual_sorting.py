import json

def test_visual_sorting():
    data = json.load(open('structuredData.json'))
    elements = [e for e in data['elements'] if 'Text' in e and e['Text'].strip()]
    
    def sort_key(elem):
        b = elem['Bounds']
        text = elem['Text'].lower()
        
        if len(b) >= 4:
            # Round Y to nearest 5 points to group similar Y positions
            y_group = round(b[1] / 5) * 5
            
            # Determine reading order priority based on content and position
            priority = 0
            
            # Address information (typically top-left, read first) - be more specific
            if any(word in text for word in ['straat', 'amsterdam', 'postcode']) or ('heer' in text and 'mevrouw' in text and len(elem['Text']) < 50 and 'geachte' not in text):
                priority = 1
            # Bank header (typically top-right, read second)
            elif any(word in text for word in ['abn amro', 'bank', 'abnamro.nl']):
                priority = 2
            # Table headers and labels (read third)
            elif any(word in text for word in ['behandeld', 'muntsoort', 'afdeling', 'leningnummer', 'datum']):
                priority = 3
            # Main content (read last) - including greetings
            else:
                priority = 4
            
            return (elem['Page'], priority, -y_group, b[0])
        return (elem['Page'], 0, 0, 0)
    
    elements.sort(key=sort_key)
    print('VISUAL READING ORDER (first 15):')
    for i, e in enumerate(elements[:15]):
        text = e['Text'].lower()
        priority = 0
        if any(word in text for word in ['straat', 'amsterdam', 'postcode']) or ('heer' in text and 'mevrouw' in text and len(e['Text']) < 50 and 'geachte' not in text):
            priority = 1
        elif any(word in text for word in ['abn amro', 'bank', 'abnamro.nl']):
            priority = 2
        elif any(word in text for word in ['behandeld', 'muntsoort', 'afdeling', 'leningnummer', 'datum']):
            priority = 3
        else:
            priority = 4
        print(f'{i+1:2d}. Priority: {priority}, Y={e["Bounds"][1]:.1f}, X={e["Bounds"][0]:.1f}: "{e["Text"][:50]}..."')

if __name__ == "__main__":
    test_visual_sorting() 