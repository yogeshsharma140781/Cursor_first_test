import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

API_KEY = os.getenv('DEEPSEEK_API_KEY')
assert API_KEY, "DEEPSEEK_API_KEY not found in environment!"

text = '''Beste ouder(s), verzorger(s), 

Donderdag 22 mei gaat de Neptunus op schoolreisje, de groepen 1 en 2 gaan naar Oud Valkeveen, www.oudvalkeveen.nl

De kinderen gaan deze dag in kleine groepjes, onder begeleiding van een leerkracht of hulpouder, het speelpark verkennen. 

 Algemene informatie:
· Alle kinderen zijn op de normale tijd in de klas. Vanaf daar lopen zij met de leerkracht en de begeleiders naar de bussen, die geparkeerd staan in de Johan van de Keukenstraat.
· Zoals u wellicht weet nemen de kinderen hun eigen eten, drinken en tussendoortje mee. Snoep mag, maar zeker geen grote zakken! 
· De kinderen mogen GEEN eigen mobiele telefoon meenemen! Dit vooral in verband met het gevaar op kwijtraken, beschadiging en de mogelijke verleiding zich uit te leven met spelletjes op de telefoon in plaats van in het pretpark.
· Zakgeld is NIET toegestaan. Het is niet leuk als het ene kind uitgebreid kan gaan shoppen, terwijl het andere geen of minder geld bij zich heeft. Ook willen we voorkomen dat de leerlingen veel tijd in de winkeltjes doorbrengen i.p.v. in het park. Het is ook niet de bedoeling dat hulpouders iets kopen voor het groepje dat ze begeleiden.
. Denkt u aan geschikte kleding (houd de weersvoorspelling in de gaten en pas de kleding hierop aan). Het is mogelijk dat de kinderen nat worden, denk aan reservekleding en een kleine handdoek.

We verwachten de kinderen, net als altijd, om 08:30 uur in de klas en de bus zal dan om 09.00 uur vertrekken. 

We vertrekken om 13.30 uur uit het park en zullen rond 14.15 uur terug zijn op school. Vanwege de te verwachten verkeersdrukte kan onze aankomsttijd mogelijk iets later uitvallen. Via Parro zullen we u hiervan op de hoogte houden.

Brengt u zelf de naschoolse opvang op de hoogte indien nodig? 

Wij hebben er zin in en maken er een mooie dag van!

Hartelijke groet, 
Team Neptunus'''

url = "https://api.deepseek.com/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}
prompt = f"""Translate the following text from Dutch to English. Only provide the translation, no explanations or additional text:\n\n{text}\n\nTranslation:"""
data = {
    "model": "deepseek-reasoner",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "temperature": 0.3,
    "max_tokens": 1024
}

response = requests.post(url, headers=headers, json=data)
response.raise_for_status()
result = response.json()
translation = result["choices"][0]["message"]["content"].strip()

print("\n--- TRANSLATION ---\n")
print(translation) 