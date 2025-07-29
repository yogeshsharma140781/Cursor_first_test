import json

def check_greeting():
    data = json.load(open('structuredData.json'))
    elements = [e for e in data['elements'] if 'Text' in e and e['Text'].strip()]
    
    def sort_key(elem):
        b = elem['Bounds']
        text = elem['Text'].lower()
        
        if len(b) >= 4:
            y_group = round(b[1] / 5) * 5
            priority = 0
            
            if any(word in text for word in ['straat', 'amsterdam', 'postcode']) or ('heer' in text and 'mevrouw' in text and len(elem['Text']) < 50 and 'geachte' not in text):
                priority = 1
            elif any(word in text for word in ['abn amro', 'bank', 'abnamro.nl']):
                priority = 2
            elif any(word in text for word in ['behandeld', 'muntsoort', 'afdeling', 'leningnummer', 'datum']):
                priority = 3
            else:
                priority = 4
            
            return (elem['Page'], priority, -y_group, b[0])
        return (elem['Page'], 0, 0, 0)
    
    elements.sort(key=sort_key)
    print('GREETING POSITION:')
    for i, e in enumerate(elements):
        if 'Geachte' in e['Text']:
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
            print(f'{i+1:2d}. Priority: {priority}, Y={e["Bounds"][1]:.1f}: "{e["Text"]}"')

if __name__ == "__main__":
    check_greeting() 