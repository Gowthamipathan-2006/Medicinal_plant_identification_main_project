"""
Medicinal Plant Monograph Database
Comprehensive botanical, phytochemical, and pharmacological knowledge base
for 40 medicinal plant species in the Ayurvedic & Ethnobotanical catalog.
"""

from typing import Dict, Any, Optional, List

MEDICINAL_PLANTS: Dict[str, Dict[str, Any]] = {
    "Aloevera": {
        "common_name": "Aloe Vera",
        "scientific_name": "Aloe barbadensis miller",
        "botanical_family": "Asphodelaceae",
        "ayurvedic_name": "Ghrita Kumari / Kanya",
        "vernacular_names": {
            "Hindi": "Ghritkumari",
            "Kannada": "Lolesara",
            "Tamil": "Kathalai",
            "Telugu": "Kalabanda",
            "Sanskrit": "Kumari"
        },
        "botanical_features": {
            "leaf_type": "Succulent, sessile rosette, lanceolate",
            "margin": "Spiny dentate with white cartilaginous teeth",
            "venation": "Parallel obscure within thick mucilaginous parenchyma",
            "leaf_apex": "Acute / spine-tipped",
            "leaf_base": "Sheathing, clasping stem"
        },
        "active_phytochemicals": [
            "Acemannan (immunomodulatory polysaccharide)",
            "Aloin A & Aloin B (barbaloin anthraquinones)",
            "Aloe-emodin",
            "Aloesin & Aloeresin",
            "Gibberellins and auxins"
        ],
        "pharmacological_actions": [
            "Epithelial wound repair and fibroblast stimulant",
            "Anti-inflammatory via thromboxane inhibition",
            "Gastroprotective and anti-ulcerogenic",
            "Topical burn soothing and UV-protective",
            "Antidiabetic glycemic control"
        ],
        "therapeutic_indications": [
            "Thermal burns, solar erythema, radiation dermatitis",
            "Gastric hyperacidity, GERD, and peptic ulcers",
            "Skin hydration, psoriasis, and wound healing",
            "Constipation (anthraquinone-rich latex)"
        ],
        "parts_used": ["Leaf gel (inner parenchyma)", "Leaf exudate (latex)"],
        "traditional_preparations": "Fresh leaf inner fillet gel, Kumari Asava (fermented Ayurvedic tonic), topical gel compresses.",
        "safety_notes": "Latex containing high aloin is a potent laxative and contraindicated in pregnancy, Crohn's disease, or kidney impairment."
    },
    "Amla": {
        "common_name": "Amla (Indian Gooseberry)",
        "scientific_name": "Phyllanthus emblica (syn. Emblica officinalis)",
        "botanical_family": "Phyllanthaceae",
        "ayurvedic_name": "Amalaki / Dhatri",
        "vernacular_names": {
            "Hindi": "Amla",
            "Kannada": "Nellikayi",
            "Tamil": "Nelli",
            "Telugu": "Usiri",
            "Sanskrit": "Amalaki"
        },
        "botanical_features": {
            "leaf_type": "Pinnate appearance, distichous, linear-oblong",
            "margin": "Entire",
            "venation": "Pinnate, subtle lateral veins",
            "leaf_apex": "Obtuse or mucronulate",
            "leaf_base": "Rounded"
        },
        "active_phytochemicals": [
            "Ascorbic acid (Vitamin C)",
            "Emblicanin A & B",
            "Gallic acid & Ellagic acid",
            "Chebulinic acid",
            "Quercetin & Kaempferol"
        ],
        "pharmacological_actions": [
            "Potent antioxidant and free-radical scavenger",
            "Rasayana (rejuvenator) and immunomodulator",
            "Hepatoprotective and lipid-lowering",
            "Gastroprotective and anti-dyspeptic",
            "Cardioprotective and anti-atherogenic"
        ],
        "therapeutic_indications": [
            "General debility, immune deficiency, and fatigue",
            "Dyspepsia, gastritis, and hyperchlorhydria",
            "Hyperlipidemia and early atherosclerosis",
            "Premature greying and hair loss"
        ],
        "parts_used": ["Fruit", "Leaves", "Bark"],
        "traditional_preparations": "Chyawanprash, Triphala Churna, Amalaki Rasayana, fresh juice with honey.",
        "safety_notes": "Extremely safe in dietary and therapeutic amounts; monitor if taking anti-platelet medications."
    },
    "Amruta_Balli": {
        "common_name": "Guduchi / Giloy (Heart-leaved Moonseed)",
        "scientific_name": "Tinospora cordifolia",
        "botanical_family": "Menispermaceae",
        "ayurvedic_name": "Guduchi / Amrita",
        "vernacular_names": {
            "Hindi": "Giloy",
            "Kannada": "Amrutha Balli",
            "Tamil": "Seenthil Kodi",
            "Telugu": "Tippa Teega",
            "Sanskrit": "Guduchi"
        },
        "botanical_features": {
            "leaf_type": "Simple, alternate, broadly cordate",
            "margin": "Entire, membranous",
            "venation": "Palmate 7-nerved at base",
            "leaf_apex": "Acuminate",
            "leaf_base": "Deeply cordate / heart-shaped"
        },
        "active_phytochemicals": [
            "Tinosporide & Cordifolide (diterpenes)",
            "Berberine (isoquinoline alkaloid)",
            "Tinocordiside & Giloin",
            "Arabino-galactan polysaccharide",
            "Magnoflorine"
        ],
        "pharmacological_actions": [
            "Immune stimulant & macrophage activator",
            "Antipyretic (Jvarahara)",
            "Hepatoprotective and detoxifying",
            "Anti-inflammatory and anti-arthritic",
            "Antidiabetic insulin secretagogue"
        ],
        "therapeutic_indications": [
            "Chronic fevers, dengue, viral infections",
            "Rheumatoid arthritis, gout (Vatarakta)",
            "Liver disorders, jaundice, hepatomegaly",
            "Allergic rhinitis and seasonal asthma"
        ],
        "parts_used": ["Stem", "Leaves", "Root"],
        "traditional_preparations": "Guduchi Satva (pure starch extract), Guduchi Kwath, Amritarishta, Samsamani Vati.",
        "safety_notes": "Well tolerated; may potentiate anti-hyperglycemic medications, monitor blood sugar levels."
    },
    "Arali": {
        "common_name": "Peepal / Sacred Fig (Arali)",
        "scientific_name": "Ficus religiosa",
        "botanical_family": "Moraceae",
        "ayurvedic_name": "Ashvattha / Pippala",
        "vernacular_names": {
            "Hindi": "Peepal",
            "Kannada": "Arali Mara",
            "Tamil": "Arasamaram",
            "Telugu": "Ravi Chettu",
            "Sanskrit": "Ashvattha"
        },
        "botanical_features": {
            "leaf_type": "Simple, alternate, broadly ovate-rotund",
            "margin": "Undulate to entire",
            "venation": "Pinnate with prominent secondary loops",
            "leaf_apex": "Caudate with long linear dripping tip (tail)",
            "leaf_base": "Cordate to truncate"
        },
        "active_phytochemicals": [
            "Beta-sitosterol",
            "Lupen-3-one",
            "Tannins and flavonoids",
            "Stigmasterol",
            "Kaempferol & Myricetin"
        ],
        "pharmacological_actions": [
            "Astringent and hemostatic (anti-hemorrhagic)",
            "Anti-inflammatory and analgesic",
            "Antidiabetic and wound healing",
            "Antibacterial and antifungal",
            "Memory enhancer and neuroprotective"
        ],
        "therapeutic_indications": [
            "Bleeding hemorrhoids and dysentery",
            "Skin ulcerations, cracked heels, eczema",
            "Asthma and bronchospasm (bark powder)",
            "Diabetes mellitus glycemic regulation"
        ],
        "parts_used": ["Bark", "Leaves", "Tender shoots", "Fruit"],
        "traditional_preparations": "Bark decoction (Kashayam), leaf juice with honey, tender leaf paste for skin.",
        "safety_notes": "Safe for standard herbal use; avoid excessively high concentrated bark extracts."
    },
    "Ashoka": {
        "common_name": "Ashoka Tree",
        "scientific_name": "Saraca asoca",
        "botanical_family": "Fabaceae (Caesalpinioideae)",
        "ayurvedic_name": "Ashoka / Gandhapushpa",
        "vernacular_names": {
            "Hindi": "Ashok",
            "Kannada": "Ashoka Mara",
            "Tamil": "Asogam",
            "Telugu": "Ashokamu",
            "Sanskrit": "Ashoka"
        },
        "botanical_features": {
            "leaf_type": "Paripinnate compound, 4-6 pairs of leaflets",
            "margin": "Entire, wavy",
            "venation": "Pinnate with fine venules",
            "leaf_apex": "Acute to acuminate",
            "leaf_base": "Rounded to cuneate"
        },
        "active_phytochemicals": [
            "Saracin & Saracasin",
            "Epicatechin & Catechin",
            "Procyanidins and condensed tannins",
            "Beta-sitosterol",
            "Quercetin"
        ],
        "pharmacological_actions": [
            "Uterine tonic and oxytocic regulator",
            "Hemostatic and astringent",
            "Anti-inflammatory and analgesic",
            "Endometriosis symptom alleviator",
            "Antimicrobial against pelvic pathogens"
        ],
        "therapeutic_indications": [
            "Menorrhagia, metrorrhagia, and dysmenorrhea",
            "Uterine fibroids and leukorrhea",
            "Hemorrhagic dysentery and internal bleeding",
            "Complexion enhancement and skin eruptions"
        ],
        "parts_used": ["Stem bark", "Flowers", "Seeds"],
        "traditional_preparations": "Ashokarishta (classical fermented tonic), Ashokakwatha with cow milk, bark decoction.",
        "safety_notes": "Avoid during pregnancy due to mild uterine contractility effects."
    },
    "Ashwagandha": {
        "common_name": "Ashwagandha (Indian Ginseng)",
        "scientific_name": "Withania somnifera",
        "botanical_family": "Solanaceae",
        "ayurvedic_name": "Ashwagandha / Balya",
        "vernacular_names": {
            "Hindi": "Ashwagandha",
            "Kannada": "Ashwagandhi",
            "Tamil": "Amukkara",
            "Telugu": "Palleru / Vajigandha",
            "Sanskrit": "Ashwagandha"
        },
        "botanical_features": {
            "leaf_type": "Simple, alternate or sub-opposite, ovate",
            "margin": "Entire, slightly undulate",
            "venation": "Pinnate, pubescent beneath",
            "leaf_apex": "Acute to obtuse",
            "leaf_base": "Cuneate"
        },
        "active_phytochemicals": [
            "Withaferin A (steroidal lactone)",
            "Withanolides A-Y",
            "Sominone & Somniferine",
            "Anaferine & Anahygrine",
            "Withanosides IV and VI"
        ],
        "pharmacological_actions": [
            "Adaptogen and cortisol-lowering agent",
            "GABA-mimetic anxiolytic and sleep promoter",
            "Neuroprotective and dendrite outgrowth stimulant",
            "Spermatogenic and testosterone booster",
            "Anti-inflammatory and muscle strength builder"
        ],
        "therapeutic_indications": [
            "Chronic stress, anxiety, insomnia, burnout",
            "Neurodegenerative decline, cognitive fog",
            "Male infertility, oligozoospermia, asthenia",
            "Sarcopenia, athletic recovery, arthritis"
        ],
        "parts_used": ["Roots", "Leaves"],
        "traditional_preparations": "Root powder (Churna) with warm milk and ghee, Ashwagandharishta, Ksmira paka.",
        "safety_notes": "Avoid in hyperthyroidism and active pregnancy unless under qualified Ayurvedic guidance."
    },
    "Avacado": {
        "common_name": "Avocado Leaf",
        "scientific_name": "Persea americana",
        "botanical_family": "Lauraceae",
        "ayurvedic_name": "Alpakathi (Modern Ethnobotany)",
        "vernacular_names": {
            "Hindi": "Makhanphal",
            "Kannada": "Benne Hannu",
            "Tamil": "Vennai Pazham",
            "Telugu": "Vennapandu",
            "Sanskrit": "Alpakathi"
        },
        "botanical_features": {
            "leaf_type": "Simple, alternate, elliptic to lanceolate",
            "margin": "Entire",
            "venation": "Pinnate, prominent secondary veins, glossy green",
            "leaf_apex": "Acute to acuminate",
            "leaf_base": "Cuneate"
        },
        "active_phytochemicals": [
            "Persin (acetogenin)",
            "Quercetin-3-glucoside",
            "Chlorogenic acid & Caffeic acid",
            "Persealactone",
            "Beta-sitosterol"
        ],
        "pharmacological_actions": [
            "Vasodilator and antihypertensive",
            "Diuretic and uric acid clearance promoter",
            "Antidiabetic and lipid-normalizing",
            "Analgesic and anti-inflammatory",
            "Gastroprotective"
        ],
        "therapeutic_indications": [
            "Essential hypertension and fluid retention",
            "Gout and elevated serum uric acid",
            "Kidney stones (nephrolithiasis prevention)",
            "Dyspepsia and mild intestinal cramps"
        ],
        "parts_used": ["Leaves", "Fruit pulp", "Seed"],
        "traditional_preparations": "Infused herbal tea (decoction of dried leaves), leaf extract.",
        "safety_notes": "Avoid in lactating animals (persin toxicity in pets); human leaf teas should be taken in moderation."
    },
    "Bamboo": {
        "common_name": "Bamboo (Bansalochan Leaf)",
        "scientific_name": "Bambusa vulgaris / Bambusa bambos",
        "botanical_family": "Poaceae",
        "ayurvedic_name": "Vamsha / Vanshlochan",
        "vernacular_names": {
            "Hindi": "Bans",
            "Kannada": "Biduru",
            "Tamil": "Moongil",
            "Telugu": "Veduru",
            "Sanskrit": "Vamsha"
        },
        "botanical_features": {
            "leaf_type": "Linear-lanceolate, alternate, short petiolate",
            "margin": "Scabrous / finely serrulate",
            "venation": "Parallel with tessellate cross-veins",
            "leaf_apex": "Acuminate",
            "leaf_base": "Rounded into pseudopetiole"
        },
        "active_phytochemicals": [
            "Organic silica (Tabasheer / Vamshalochana)",
            "Orientin & Iso-orientin",
            "Vitexin & Isovitexin",
            "Phenolic acids",
            "Flavone glycosides"
        ],
        "pharmacological_actions": [
            "Bone remineralization & collagen collagen-builder",
            "Antipyretic and cooling demulcent",
            "Antioxidant and vascular strengthener",
            "Expectorant and anti-inflammatory",
            "Emmenagogue (in folk traditions)"
        ],
        "therapeutic_indications": [
            "Osteoporosis, connective tissue weakness, joint pain",
            "Cough, bronchitis, respiratory fever",
            "Bleeding gums and internal heat (Pitta excess)",
            "Skin fragility and brittle nails"
        ],
        "parts_used": ["Leaf extract", "Siliceous concretions (Vanshalochan)", "Young shoots"],
        "traditional_preparations": "Sitopaladi Churna (containing Tabasheer), leaf infusion, shoot extract.",
        "safety_notes": "Raw shoots contain cyanogenic glycosides and must be boiled before culinary use. Leaf extracts are gentle."
    },
    "Basale": {
        "common_name": "Malabar Spinach (Basale)",
        "scientific_name": "Basella alba (syn. B. rubra)",
        "botanical_family": "Basellaceae",
        "ayurvedic_name": "Upodika / Potaki",
        "vernacular_names": {
            "Hindi": "Poi Saag",
            "Kannada": "Basale Soppu",
            "Tamil": "Pasalai Keerai",
            "Telugu": "Bachalikura",
            "Sanskrit": "Upodika"
        },
        "botanical_features": {
            "leaf_type": "Simple, alternate, succulent, cordate-ovate",
            "margin": "Entire, thickened",
            "venation": "Pinnate, fleshy succulent veins",
            "leaf_apex": "Obtuse or acute",
            "leaf_base": "Cordate"
        },
        "active_phytochemicals": [
            "Mucilage (arabinogalactan polysaccharides)",
            "Betalains (gomphrenin & betanin in rubra)",
            "Lutein & Zeaxanthin",
            "Ascorbic acid and Vitamin A precursor",
            "Iron, calcium, and zinc"
        ],
        "pharmacological_actions": [
            "Demulcent, gastroprotective, and soothing",
            "Mild cooling laxative and stool softener",
            "Wound healing and anti-ulcerogenic",
            "Aphrodisiac and nourishing tonic (Balya)",
            "Antioxidant and anti-inflammatory"
        ],
        "therapeutic_indications": [
            "Gastric ulcers, aphthous stomatitis (mouth ulcers)",
            "Chronic constipation and hemorrhoidal irritation",
            "Anemia and nutritional convalescence",
            "Pruritus and inflammatory skin rashes"
        ],
        "parts_used": ["Leaves", "Fleshy stems"],
        "traditional_preparations": "Fresh leaf juice applied to ulcers, cooked botanical potherb, mucilaginous poultice.",
        "safety_notes": "High soluble oxalates; consume cooked and moderate intake if prone to calcium-oxalate kidney stones."
    },
    "Betel": {
        "common_name": "Betel Leaf (Paan)",
        "scientific_name": "Piper betle",
        "botanical_family": "Piperaceae",
        "ayurvedic_name": "Tambula / Nagavalli",
        "vernacular_names": {
            "Hindi": "Paan",
            "Kannada": "Vilyadele",
            "Tamil": "Vetrillai",
            "Telugu": "Tamalapaku",
            "Sanskrit": "Tambula"
        },
        "botanical_features": {
            "leaf_type": "Simple, alternate, broadly ovate-cordate",
            "margin": "Entire",
            "venation": "Palmate 5-7 curving nerved from base",
            "leaf_apex": "Acuminate",
            "leaf_base": "Oblique cordate"
        },
        "active_phytochemicals": [
            "Chavibetol & Chavicol (allylphenols)",
            "Eugenol & Eugenol methyl ether",
            "Hydroxychavicol",
            "Beta-caryophyllene",
            "Piperitol and terpinen-4-ol"
        ],
        "pharmacological_actions": [
            "Potent oral antiseptic and plaque inhibitor",
            "Carminative and digestive stimulant (Deepana-Pachana)",
            "Antioxidant and free radical neutralizer",
            "Topical anti-inflammatory and analgesic",
            "Bronchodilator and mucolytic"
        ],
        "therapeutic_indications": [
            "Halitosis, periodontal infection, and gingivitis",
            "Indigestion, dyspepsia, and flatulence",
            "Cough, respiratory congestion (warmed leaf application)",
            "Minor cuts, burns, and localized boils"
        ],
        "parts_used": ["Fresh leaves", "Essential oil"],
        "traditional_preparations": "Paan with digestive spices, fresh leaf juice (Swarasa) with honey, warm leaf castor oil compress.",
        "safety_notes": "Medicinal pure leaf is highly beneficial; avoid commercial habit of chewing with tobacco or slaked lime."
    },
    "Betel_Nut": {
        "common_name": "Areca Nut / Betel Nut Palm",
        "scientific_name": "Areca catechu",
        "botanical_family": "Arecaceae",
        "ayurvedic_name": "Puga / Kramuka",
        "vernacular_names": {
            "Hindi": "Supari",
            "Kannada": "Adike",
            "Tamil": "Pakku",
            "Telugu": "Vakka",
            "Sanskrit": "Puga"
        },
        "botanical_features": {
            "leaf_type": "Pinnately compound fronds, 1-1.5m long, crowded leaflets",
            "margin": "Entire, linear-lanceolate",
            "venation": "Parallel along leaflet axis",
            "leaf_apex": "Bifid or praemorse",
            "leaf_base": "Sheathing"
        },
        "active_phytochemicals": [
            "Arecoline (cholinergic muscarinic alkaloid)",
            "Arecaidine, Guvacine, Guvacoline",
            "Condensed tannins and procyanidins",
            "Catechin and epicatechin",
            "Gallic acid derivatives"
        ],
        "pharmacological_actions": [
            "Parasympathomimetic and salivatory stimulant",
            "Anthelmintic (expels intestinal tapeworms)",
            "Astringent and hemostatic",
            "Mild stimulant and alertness enhancer",
            "Antimicrobial against dental microbes"
        ],
        "therapeutic_indications": [
            "Intestinal helminthiasis (tapeworm infection)",
            "Gingival bleeding and laxity (astringent mouth rinses)",
            "Digestive sluggishness (in micro-doses)",
            "Excessive salivation or diarrhea in classical formulas"
        ],
        "parts_used": ["Nut (seed)", "Frond leaves"],
        "traditional_preparations": "Puga Khanda (classical Ayurvedic confection), processed nut decoction for helminths.",
        "safety_notes": "Chronic daily chewing of raw nut is strongly associated with oral submucous fibrosis; therapeutic use must be strictly medicinal."
    },
    "Brahmi": {
        "common_name": "Brahmi / Water Hyssop",
        "scientific_name": "Bacopa monnieri",
        "botanical_family": "Plantaginaceae",
        "ayurvedic_name": "Brahmi / Medhya Rasayana",
        "vernacular_names": {
            "Hindi": "Brahmi",
            "Kannada": "Brahmi Soppu / Ondelaga",
            "Tamil": "Neerbrahmi",
            "Telugu": "Sambhramani",
            "Sanskrit": "Brahmi"
        },
        "botanical_features": {
            "leaf_type": "Simple, opposite, succulent, sessile, obovate-oblong",
            "margin": "Entire",
            "venation": "Obscure pinnate with glandular dots",
            "leaf_apex": "Obtuse / rounded",
            "leaf_base": "Cuneate"
        },
        "active_phytochemicals": [
            "Bacoside A & B (dammarane triterpenoid saponins)",
            "Bacopasides I-XII",
            "Bacopasaponins",
            "Luteolin & Apigenin",
            "Hersaponin and Betulinic acid"
        ],
        "pharmacological_actions": [
            "Medhya Rasayana (synaptic plasticity & memory enhancer)",
            "Anxiolytic and adaptogen via serotonergic modulation",
            "Antioxidant in striatum and hippocampus",
            "Neuroprotective against beta-amyloid aggregation",
            "Anticonvulsant and attention booster"
        ],
        "therapeutic_indications": [
            "Memory loss, ADHD, age-related cognitive impairment",
            "Generalized anxiety, mental fatigue, exam stress",
            "Epilepsy (Apasmara adjuvant support)",
            "Insomnia and nervous restlessness"
        ],
        "parts_used": ["Whole herb", "Fresh leaves"],
        "traditional_preparations": "Brahmi Ghrita (medicated clarified butter), Saraswatarishta, Brahmi Churna, fresh leaf chutney.",
        "safety_notes": "Safe and non-sedating; occasionally causes mild gastrointestinal cramping on an empty stomach."
    },
    "Castor": {
        "common_name": "Castor Plant (Eranda)",
        "scientific_name": "Ricinus communis",
        "botanical_family": "Euphorbiaceae",
        "ayurvedic_name": "Eranda / Gandharvahasta",
        "vernacular_names": {
            "Hindi": "Arandi",
            "Kannada": "Oudala",
            "Tamil": "Amanakku",
            "Telugu": "Amudamu",
            "Sanskrit": "Eranda"
        },
        "botanical_features": {
            "leaf_type": "Large, alternate, peltate, palmately 7-11 lobed",
            "margin": "Serrate to glandular-toothed",
            "venation": "Palmate, radiating strongly from petiole insertion",
            "leaf_apex": "Acuminate",
            "leaf_base": "Peltate / circular insertion"
        },
        "active_phytochemicals": [
            "Ricinoleic acid (hydroxy fatty acid in seed oil)",
            "Ricinidine (alkaloid in leaves)",
            "Rutin & Hyperoside",
            "Kaempferol-3-rutinoside",
            "Gallic acid"
        ],
        "pharmacological_actions": [
            "Potent anti-inflammatory & anti-arthritic (Vatahara)",
            "Stimulant laxative (ricinoleic acid EP3 receptor agonist)",
            "Analgesic and nerve conduction supporter",
            "Galactagogue (leaf warm application)",
            "Topical wound and muscle relaxation promoter"
        ],
        "therapeutic_indications": [
            "Rheumatoid arthritis, osteoarthritis, sciatica, lumbago",
            "Acute constipation and bowel cleansing",
            "Lactation insufficiency (warm leaf breast compresses)",
            "Skin calluses and abdominal distension"
        ],
        "parts_used": ["Root", "Leaves", "Cold-pressed seed oil"],
        "traditional_preparations": "Erandapak, Gandharvahasthadi Kashayam, Eranda Taila (castor oil), leaf poultices.",
        "safety_notes": "Castor seed hull contains toxic ricin protein; medicinal castor oil is cold-pressed and toxic ricin is destroyed. Leaves are applied topically or boiled."
    },
    "Curry_Leaf": {
        "common_name": "Curry Leaf (Kadi Patta)",
        "scientific_name": "Murraya koenigii",
        "botanical_family": "Rutaceae",
        "ayurvedic_name": "Girinimba / Kalashaka",
        "vernacular_names": {
            "Hindi": "Kadi Patta",
            "Kannada": "Karibevu",
            "Tamil": "Karuveppilai",
            "Telugu": "Karivepaku",
            "Sanskrit": "Girinimba"
        },
        "botanical_features": {
            "leaf_type": "Pinnately compound, 11-25 alternate leaflets, ovate-lanceolate",
            "margin": "Slightly dentate / crenulate, gland-dotted",
            "venation": "Pinnate, translucent pellucid oil glands",
            "leaf_apex": "Acuminate",
            "leaf_base": "Asymmetric / oblique"
        },
        "active_phytochemicals": [
            "Mahanimbine & Girinimbine (carbazole alkaloids)",
            "Koenimbine & Mahanine",
            "Beta-caryophyllene & Alpha-pinene",
            "Lutein & Beta-carotene",
            "Rutin and Quercetin"
        ],
        "pharmacological_actions": [
            "Hypoglycemic & alpha-glucosidase inhibitor",
            "Hypolipidemic and cholesterol-reducing",
            "Antioxidant and hepatoprotective",
            "Hair melanogenesis stimulator (prevents greying)",
            "Digestive carminative and anti-dysenteric"
        ],
        "therapeutic_indications": [
            "Type 2 diabetes glycemic control",
            "Hypercholesterolemia and non-alcoholic fatty liver",
            "Hair thinning, premature greying, alopecia",
            "Morning sickness, diarrhea, dyspepsia"
        ],
        "parts_used": ["Fresh leaves", "Bark", "Roots"],
        "traditional_preparations": "Fresh leaf juice with buttermilk, medicated hair oils (boiled in coconut oil), culinary daily seasoning.",
        "safety_notes": "Extremely safe; suitable for long-term daily dietary consumption."
    },
    "Doddapatre": {
        "common_name": "Indian Borage / Mexican Mint (Doddapatre)",
        "scientific_name": "Plectranthus amboinicus (syn. Coleus amboinicus)",
        "botanical_family": "Lamiaceae",
        "ayurvedic_name": "Parna Yavano / Pashanabhedi",
        "vernacular_names": {
            "Hindi": "Patta Ajwain",
            "Kannada": "Doddapatre / Sambrani Soppu",
            "Tamil": "Karpuravalli",
            "Telugu": "Vamu Aku",
            "Sanskrit": "Karpuravalli"
        },
        "botanical_features": {
            "leaf_type": "Simple, opposite, thick succulent, broadly ovate",
            "margin": "Coarsely crenate to dentate",
            "venation": "Reticulate, densely pubescent velvety surface",
            "leaf_apex": "Obtuse to acute",
            "leaf_base": "Cuneate to truncate"
        },
        "active_phytochemicals": [
            "Carvacrol (dominant phenolic volatile)",
            "Thymol & Eugenol",
            "Rosmarinic acid",
            "Caryophyllene",
            "Quercetin & Apigenin"
        ],
        "pharmacological_actions": [
            "Potent bronchodilator and mucolytic",
            "Antibacterial and antifungal against respiratory bugs",
            "Carminative and antispasmodic",
            "Diuretic and antilithiatic (kidney stone dissolver)",
            "Anti-inflammatory and antipyretic"
        ],
        "therapeutic_indications": [
            "Pediatric cough, cold, bronchitis, sore throat",
            "Renal calculi (small kidney stones) and dysuria",
            "Infantile colic, abdominal gas, indigestion",
            "Allergic asthma and congested sinuses"
        ],
        "parts_used": ["Fresh succulent leaves"],
        "traditional_preparations": "Freshly extracted leaf juice with honey, roasted leaf juice for infants, Doddapatre Tambuli (yoghurt dish).",
        "safety_notes": "Very safe for all age groups including infants in traditional spoonful dosages."
    },
    "Ekka": {
        "common_name": "Crown Flower / Giant Milkweed (Ekka)",
        "scientific_name": "Calotropis gigantea",
        "botanical_family": "Apocynaceae (Asclepiadoideae)",
        "ayurvedic_name": "Arka / Mandara",
        "vernacular_names": {
            "Hindi": "Aak / Madar",
            "Kannada": "Ekka / Yekke",
            "Tamil": "Erukku",
            "Telugu": "Jilledu",
            "Sanskrit": "Arka"
        },
        "botanical_features": {
            "leaf_type": "Simple, opposite decussate, sessile, obovate-oblong",
            "margin": "Entire, leathery thick",
            "venation": "Pinnate, covered with white cottony tomentum",
            "leaf_apex": "Acute to short acuminate",
            "leaf_base": "Cordate / auriculate"
        },
        "active_phytochemicals": [
            "Calotropin & Calotoxin (cardenolide glycosides)",
            "Uscharin & Giganteol",
            "Alpha & Beta-amyrin",
            "Calotropenyl acetate",
            "Latex proteolytic enzymes"
        ],
        "pharmacological_actions": [
            "Potent anti-inflammatory and rubefacient",
            "Analgesic for musculoskeletal pain",
            "Bronchodilator and expectorant (micro-doses)",
            "Wound debridement and antimicrobial",
            "Anthelmintic and purgative"
        ],
        "therapeutic_indications": [
            "Rheumatic joint swelling, osteoarthritic knee pain",
            "Sciatica, muscular sprains, deep contusions",
            "Skin infections, non-healing chronic ulcers",
            "Bronchial asthma and severe cough (special Ayurvedic detoxed formulations)"
        ],
        "parts_used": ["Leaves", "Roots", "Latex", "Flowers"],
        "traditional_preparations": "Warmed leaves smeared with castor oil tied over painful joints, Arka Taila, Arkalavana.",
        "safety_notes": "Latex is toxic and highly irritating to eyes; internal administration requires classical Ayurvedic purification (Shodhana)."
    },
    "Ganike": {
        "common_name": "Black Nightshade (Ganike)",
        "scientific_name": "Solanum nigrum",
        "botanical_family": "Solanaceae",
        "ayurvedic_name": "Kakamachi / Kakadani",
        "vernacular_names": {
            "Hindi": "Makoy",
            "Kannada": "Ganike Soppu / Kage Soppu",
            "Tamil": "Manathakkali Keerai",
            "Telugu": "Kamanchi",
            "Sanskrit": "Kakamachi"
        },
        "botanical_features": {
            "leaf_type": "Simple, alternate, ovate to elliptic",
            "margin": "Sinuately dentate or lobed to entire",
            "venation": "Pinnate",
            "leaf_apex": "Acute to acuminate",
            "leaf_base": "Attenuate into petiole"
        },
        "active_phytochemicals": [
            "Solamargine & Solasonine (steroidal glycoalkaloids)",
            "Solasodine",
            "Anthocyanidins & Polyphenols",
            "Beta-carotene",
            "Gallic acid and Rutin"
        ],
        "pharmacological_actions": [
            "Hepatoprotective and liver regenerator",
            "Anti-ulcerogenic and mucosal protective",
            "Cardiotonic and mild diuretic",
            "Anti-inflammatory and antipyretic",
            "Cytoprotective against drug-induced toxicity"
        ],
        "therapeutic_indications": [
            "Hepatitis, liver cirrhosis, jaundice",
            "Gastric ulcers and stomatitis (mouth ulcers)",
            "Edema, chronic skin diseases, pruritus",
            "Fever and splenomegaly"
        ],
        "parts_used": ["Whole plant", "Leaves", "Berries"],
        "traditional_preparations": "Fresh leaf juice (Swarasa), cooked leaf soup/porridge for mouth ulcers, Kakamachyadi Ghrita.",
        "safety_notes": "Ripe dark berries and cooked leaves are medicinal; avoid large quantities of raw unripe green berries due to solanine."
    },
    "Gauva": {
        "common_name": "Guava Leaf",
        "scientific_name": "Psidium guajava",
        "botanical_family": "Myrtaceae",
        "ayurvedic_name": "Amrita Phalam / Peruka",
        "vernacular_names": {
            "Hindi": "Amrood",
            "Kannada": "Seebe Soppu / Perale",
            "Tamil": "Koyya Ilai",
            "Telugu": "Jama Aaku",
            "Sanskrit": "Peruka"
        },
        "botanical_features": {
            "leaf_type": "Simple, opposite, elliptic-oblong, aromatic when crushed",
            "margin": "Entire",
            "venation": "Pinnate with prominent parallel secondary veins embossed beneath",
            "leaf_apex": "Obtuse to acute",
            "leaf_base": "Rounded"
        },
        "active_phytochemicals": [
            "Guajaverin & Quercetin-3-O-galactoside",
            "Caryophyllene & Cineole",
            "Gallic acid & Ellagic acid",
            "Avicularin",
            "Condensed tannins"
        ],
        "pharmacological_actions": [
            "Potent antidiarrheal and gastrointestinal astringent",
            "Antidiabetic alpha-amylase and alpha-glucosidase inhibitor",
            "Antimicrobial against oral and enteric pathogens",
            "Anti-inflammatory and toothache reliever",
            "Cardioprotective and hypotensive"
        ],
        "therapeutic_indications": [
            "Acute watery diarrhea, dysentery, gastroenteritis",
            "Postprandial hyperglycemia and diabetes management",
            "Toothache, swollen bleeding gums, mouth ulcers (gargle)",
            "Superficial wound antiseptic washes"
        ],
        "parts_used": ["Young tender leaves", "Bark", "Fruit"],
        "traditional_preparations": "Tender leaf decoction (tea), chewed raw leaf for toothache, leaf mouthwash rinse.",
        "safety_notes": "Very safe; excessive high concentrated intake may cause temporary constipation."
    },
    "Geranium": {
        "common_name": "Rose Geranium",
        "scientific_name": "Pelargonium graveolens",
        "botanical_family": "Geraniaceae",
        "ayurvedic_name": "Sugandha Patra (Aromatherapeutic)",
        "vernacular_names": {
            "Hindi": "Geranium",
            "Kannada": "Geranium Soppu",
            "Tamil": "Geranium Ilai",
            "Telugu": "Geranium",
            "Sanskrit": "Sugandhapatra"
        },
        "botanical_features": {
            "leaf_type": "Alternate, palmately deeply 5-7 lobed, velvety glandular",
            "margin": "Dentate to pinnatifid, highly rose-scented",
            "venation": "Palmate",
            "leaf_apex": "Acute",
            "leaf_base": "Cordate"
        },
        "active_phytochemicals": [
            "Citronellol & Geraniol (terpene alcohols)",
            "Linalool & Citronellyl formate",
            "Isomenthone",
            "Geranyl butyrate",
            "Flavonoid glycosides"
        ],
        "pharmacological_actions": [
            "Anxiolytic and neuro-sedative aromatherapeutic",
            "Antibacterial, antifungal, and antiseptic",
            "Sebum-balancing and dermatological astringent",
            "Anti-inflammatory and wound healer",
            "Circulatory stimulant and lymphatic tonic"
        ],
        "therapeutic_indications": [
            "Nervous tension, emotional anxiety, insomnia",
            "Acne vulgaris, eczema, oily dermatitis",
            "Minor burns, cuts, insect bites",
            "Menopausal hot flashes and hormonal mood swings"
        ],
        "parts_used": ["Leaves", "Essential oil"],
        "traditional_preparations": "Steam-distilled essential oil, dried leaf infusion, herbal bath teas.",
        "safety_notes": "Dilute essential oil in carrier oil before skin application; safe for aromatic inhalation."
    },
    "Henna": {
        "common_name": "Henna (Mehndi)",
        "scientific_name": "Lawsonia inermis",
        "botanical_family": "Lythraceae",
        "ayurvedic_name": "Madayanthika / Mendhi",
        "vernacular_names": {
            "Hindi": "Mehndi",
            "Kannada": "Goranti Soppu / Mehendi",
            "Tamil": "Maruthani",
            "Telugu": "Gorintaku",
            "Sanskrit": "Madayanthika"
        },
        "botanical_features": {
            "leaf_type": "Simple, opposite, subsessile, elliptic to lanceolate",
            "margin": "Entire",
            "venation": "Pinnate with depressed veins",
            "leaf_apex": "Acute to acuminate",
            "leaf_base": "Cuneate"
        },
        "active_phytochemicals": [
            "Lawsone (2-hydroxy-1,4-naphthoquinone - natural red-orange dye)",
            "Galloyl glucoside & Tannins",
            "Luteolin & Apigenin",
            "Beta-sitosterol",
            "Flavonoid glycosides"
        ],
        "pharmacological_actions": [
            "Cooling antipyretic and refrigerant (Daha Prashamana)",
            "Broad-spectrum antifungal (especially Tinea / Candida)",
            "Keratin binding and natural hair conditioner",
            "Astringent and wound cicatrizing",
            "Anti-inflammatory and antimicrobial"
        ],
        "therapeutic_indications": [
            "Burning sensations in palms and soles (burning feet syndrome)",
            "Tinea pedis (athlete's foot), ringworm, fungal scalp dandruff",
            "Hair breakage, premature greying, scalp cooling",
            "Mild skin burns, prickly heat, and eczema"
        ],
        "parts_used": ["Leaves", "Bark", "Flowers", "Seeds"],
        "traditional_preparations": "Fresh leaf paste applied to palms/soles, dried leaf powder hair pack, Henna oil.",
        "safety_notes": "Natural pure henna is extremely safe; avoid synthetic 'black henna' adulterated with toxic PPD (paraphenylenediamine)."
    },
    "Hibiscus": {
        "common_name": "Hibiscus / China Rose (Gudhal)",
        "scientific_name": "Hibiscus rosa-sinensis",
        "botanical_family": "Malvaceae",
        "ayurvedic_name": "Japa / Java",
        "vernacular_names": {
            "Hindi": "Gudhal",
            "Kannada": "Dasavala",
            "Tamil": "Sembaruthi",
            "Telugu": "Mandara",
            "Sanskrit": "Japa"
        },
        "botanical_features": {
            "leaf_type": "Simple, alternate, ovate to elliptic",
            "margin": "Serrate or dentate towards upper half",
            "venation": "Palmate at base transitioning to pinnate",
            "leaf_apex": "Acute to acuminate",
            "leaf_base": "Cuneate to rounded"
        },
        "active_phytochemicals": [
            "Anthocyanins (Cyanidin-3-sophoroside)",
            "Flavonoids (Quercetin, Kaempferol)",
            "Hibiscic acid & Citric acid",
            "Mucilage polysaccharides",
            "Beta-sitosterol and phytosterols"
        ],
        "pharmacological_actions": [
            "Hair follicle anagen stimulator & anti-dandruff",
            "Antihypertensive (mild ACE inhibitor effect)",
            "Hypolipidemic and antioxidant",
            "Demulcent and cooling refrigerant",
            "Emmenagogue and menstrual regulator"
        ],
        "therapeutic_indications": [
            "Alopecia, premature greying, hair thinning",
            "Mild essential hypertension",
            "Hypercholesterolemia and lipid oxidation",
            "Dysuria, cystitis, and hot flashes"
        ],
        "parts_used": ["Leaves", "Flowers", "Petals"],
        "traditional_preparations": "Leaf mucilage paste for hair conditioning, Hibiscus petal sour tea, cold-infused hair oil.",
        "safety_notes": "High doses of flower/leaf extracts possess mild antifertility activity; caution in early pregnancy."
    },
    "Honge": {
        "common_name": "Pongamia / Indian Beech (Karanja)",
        "scientific_name": "Pongamia pinnata (syn. Millettia pinnata)",
        "botanical_family": "Fabaceae",
        "ayurvedic_name": "Karanja / Naktamala",
        "vernacular_names": {
            "Hindi": "Karanj",
            "Kannada": "Honge Mara",
            "Tamil": "Pungai",
            "Telugu": "Kanuga",
            "Sanskrit": "Karanja"
        },
        "botanical_features": {
            "leaf_type": "Imparipinnate compound, 5-7 ovate-elliptic leaflets",
            "margin": "Entire",
            "venation": "Pinnate, glossy glabrous dark green surface",
            "leaf_apex": "Short acuminate",
            "leaf_base": "Rounded to cuneate"
        },
        "active_phytochemicals": [
            "Karanjin (furanoflavonol)",
            "Pongamol & Pinnatin",
            "Pongapin & Kanjone",
            "Beta-sitosterol",
            "Fatty acids (oleic & linoleic acid)"
        ],
        "pharmacological_actions": [
            "Potent anthelmintic, insecticidal, and antiparasitic",
            "Broad-spectrum antimicrobial and antifungal",
            "Anti-inflammatory and analgesic (Kushtaghna)",
            "Wound cicatrizing and ulcer cleanser (Vranaropana)",
            "Antidiabetic and hypoglycemic"
        ],
        "therapeutic_indications": [
            "Chronic skin diseases, scabies, eczema, psoriasis",
            "Rheumatic joint swelling and arthritis (Karanja oil massage)",
            "Infected non-healing wounds and diabetic ulcers",
            "Dental caries, bleeding gums (twigs used as datun)"
        ],
        "parts_used": ["Leaves", "Seeds / Seed oil (Karanja Taila)", "Bark", "Roots"],
        "traditional_preparations": "Karanja Taila for external skin application, leaf decoction wash for wounds, Jatyadi Taila constituent.",
        "safety_notes": "Karanja seed oil is non-edible and intended exclusively for topical external use. Leaf extracts used topically."
    },
    "Insulin": {
        "common_name": "Insulin Plant / Fiery Costus",
        "scientific_name": "Chamaecostus cuspidatus (syn. Costus igneus)",
        "botanical_family": "Costaceae",
        "ayurvedic_name": "Madhura Valli (Modern Ethnobotany)",
        "vernacular_names": {
            "Hindi": "Insulin Plant",
            "Kannada": "Insulin Soppu",
            "Tamil": "Insulin Chedi",
            "Telugu": "Insulin Aaku",
            "Sanskrit": "Kostha"
        },
        "botanical_features": {
            "leaf_type": "Simple, spirally arranged along reddish stems, oblong-lanceolate",
            "margin": "Entire, wavy",
            "venation": "Parallel curving from prominent midrib",
            "leaf_apex": "Acuminate",
            "leaf_base": "Cuneate"
        },
        "active_phytochemicals": [
            "Corosolic acid (triterpene - promotes cellular glucose uptake)",
            "Diosgenin (steroidal sapogenin)",
            "Beta-carotene & Alpha-tocopherol",
            "Lupeol & Stigmasterol",
            "Quercetin and Kaempferol"
        ],
        "pharmacological_actions": [
            "Potent antidiabetic & insulin sensitizer",
            "Beta-cell protective and regenerator in islets of Langerhans",
            "Hypolipidemic (lowers triglycerides & LDL)",
            "Antioxidant and anti-inflammatory",
            "Diuretic and antimicrobial"
        ],
        "therapeutic_indications": [
            "Type 2 diabetes mellitus glycemic management",
            "Insulin resistance and metabolic syndrome",
            "Hyperlipidemia in diabetic patients",
            "Diabetic neuropathy prevention"
        ],
        "parts_used": ["Fresh leaves", "Dried leaf powder"],
        "traditional_preparations": "Chewing 1-2 fresh leaves daily morning on empty stomach, dried leaf powder with warm water.",
        "safety_notes": "Monitor blood glucose closely to prevent hypoglycemia when combined with prescription antidiabetic drugs."
    },
    "Jasmine": {
        "common_name": "Jasmine (Mogra / Chameli)",
        "scientific_name": "Jasminum sambac / Jasminum officinale",
        "botanical_family": "Oleaceae",
        "ayurvedic_name": "Mallika / Jati",
        "vernacular_names": {
            "Hindi": "Mogra / Chameli",
            "Kannada": "Malli / Mallige Soppu",
            "Tamil": "Malli",
            "Telugu": "Malle Puvvu",
            "Sanskrit": "Mallika"
        },
        "botanical_features": {
            "leaf_type": "Opposite or in whorls of 3, simple or trifoliolate, ovate",
            "margin": "Entire, slightly undulate",
            "venation": "Pinnate with arched lateral nerves",
            "leaf_apex": "Acute to obtuse",
            "leaf_base": "Rounded to cuneate"
        },
        "active_phytochemicals": [
            "Benzyl acetate & Linalool (fragrance esters)",
            "Jasmonic acid & Methyl jasmonate",
            "Hesperidin & Rutin",
            "Sambacosides A-F",
            "Secoiridoid glucosides"
        ],
        "pharmacological_actions": [
            "Sedative, antidepressant, and anxiolytic",
            "Lactation suppressor (when applied topically to breasts)",
            "Antiseptic, wound cicatrizing, and cooling",
            "Antipyretic and anti-inflammatory (Pitta pacifier)",
            "Antioxidant and analgesic"
        ],
        "therapeutic_indications": [
            "Emotional anxiety, tension headaches, insomnia",
            "Weaning breast engorgement / galactostasis",
            "Mouth ulcers, stomatitis, ocular burning",
            "Skin pruritus, sunburn, and skin irritation"
        ],
        "parts_used": ["Flowers", "Leaves", "Essential oil"],
        "traditional_preparations": "Jasmine flower tea, fresh leaf gargle for mouth ulcers, Jatyadi Taila for chronic wounds.",
        "safety_notes": "Safe and non-toxic; gentle cooling action suitable for regular use."
    },
    "Lemon": {
        "common_name": "Lemon Leaf",
        "scientific_name": "Citrus limon",
        "botanical_family": "Rutaceae",
        "ayurvedic_name": "Nimbuka / Jambira",
        "vernacular_names": {
            "Hindi": "Nimbu",
            "Kannada": "Nimbe Soppu",
            "Tamil": "Elumichai Ilai",
            "Telugu": "Nimma Aaku",
            "Sanskrit": "Jambira"
        },
        "botanical_features": {
            "leaf_type": "Unifoliolate compound, alternate, elliptic-ovate",
            "margin": "Crenulate to serrulate, translucent oil glands",
            "venation": "Pinnate, winged or margined petiole",
            "leaf_apex": "Acute",
            "leaf_base": "Rounded"
        },
        "active_phytochemicals": [
            "Limonene & Citral (geranial & neral)",
            "Hesperidin & Eriocitrin (flavanone glycosides)",
            "Linalool & Alpha-terpineol",
            "Citric acid",
            "Vitamin C and bioflavonoids"
        ],
        "pharmacological_actions": [
            "Antimicrobial, antiviral, and antiseptic",
            "Digestive stimulant and anti-nausea",
            "Anxiolytic and mood uplifter via olfactory pathways",
            "Antioxidant and vascular capillary protectant",
            "Febrifuge and diaphoretic (promotes sweating in fever)"
        ],
        "therapeutic_indications": [
            "Nausea, travel sickness, morning sickness, indigestion",
            "Viral colds, sore throat, feverish chills (steam inhalation)",
            "Mild anxiety, stress, mental fatigue",
            "Oily skin, dandruff, and scalp clarification"
        ],
        "parts_used": ["Leaves", "Fruit juice", "Peel oil"],
        "traditional_preparations": "Crushed leaf herbal infusion (tea), steam inhalation with boiled leaves, culinary citrus flavoring.",
        "safety_notes": "Very safe; citrus leaf oils may cause mild photosensitivity if applied concentrated on skin exposed to intense sunlight."
    },
    "Lemon_grass": {
        "common_name": "Lemongrass",
        "scientific_name": "Cymbopogon citratus",
        "botanical_family": "Poaceae",
        "ayurvedic_name": "Bhutrina / Rohisha",
        "vernacular_names": {
            "Hindi": "Nimbu Ghas",
            "Kannada": "Majjige Hullu",
            "Tamil": "Elumichai Pul",
            "Telugu": "Nimmagaddi",
            "Sanskrit": "Bhutrina"
        },
        "botanical_features": {
            "leaf_type": "Linear, elongated strap-like grass blades, up to 1m long",
            "margin": "Scabrous / rough, sharply cutting",
            "venation": "Parallel with prominent white central midrib",
            "leaf_apex": "Attenuate / acuminate",
            "leaf_base": "Sheathing"
        },
        "active_phytochemicals": [
            "Citral (geranial & neral, 70-80% of volatile oil)",
            "Myrcene & Geraniol",
            "Luteolin & Isoorientin",
            "Chlorogenic acid & Caffeic acid",
            "Cymbopogonol"
        ],
        "pharmacological_actions": [
            "Antipyretic and diaphoretic (fever reducer)",
            "Broad-spectrum antimicrobial and antifungal",
            "Carminative, stomachic, and anti-spasmodic",
            "Anxiolytic and mild sedative",
            "Insect repellent (mosquitoes)"
        ],
        "therapeutic_indications": [
            "Feverish colds, flu, upper respiratory congestion",
            "Digestive cramps, flatulence, bloating",
            "Rheumatic muscle pain (topical massage oil)",
            "Insomnia, nervous headache, anxiety"
        ],
        "parts_used": ["Fresh and dried aerial grass blades"],
        "traditional_preparations": "Lemongrass herbal decoction (tea), essential oil in carrier oil, culinary Thai/Ayurvedic soups.",
        "safety_notes": "Safe in culinary and herbal tea amounts; avoid high-dose essential oil ingestion during pregnancy."
    },
    "Mango": {
        "common_name": "Mango Leaf",
        "scientific_name": "Mangifera indica",
        "botanical_family": "Anacardiaceae",
        "ayurvedic_name": "Amra / Rasala",
        "vernacular_names": {
            "Hindi": "Aam",
            "Kannada": "Mavina Soppu / Mavu",
            "Tamil": "Maa Ilai",
            "Telugu": "Mamidikura",
            "Sanskrit": "Amra"
        },
        "botanical_features": {
            "leaf_type": "Simple, alternate, oblong-lanceolate, leathery",
            "margin": "Entire, undulate",
            "venation": "Pinnate with prominent parallel lateral secondary veins",
            "leaf_apex": "Acuminate",
            "leaf_base": "Cuneate"
        },
        "active_phytochemicals": [
            "Mangiferin (C-glucosylxanthone - powerful antioxidant)",
            "Iso-mangiferin",
            "Gallic acid & Ellagic acid",
            "Flavonoids (Hyperoside, Quercetin)",
            "Tannins and triterpenoids"
        ],
        "pharmacological_actions": [
            "Antidiabetic (PPAR-gamma activation & glucose uptake)",
            "Antioxidant and radioprotective",
            "Antihypertensive and hypotensive",
            "Gastroprotective and anti-ulcerogenic",
            "Anti-inflammatory and antimicrobial"
        ],
        "therapeutic_indications": [
            "Early stage type 2 diabetes and diabetic angiopathy",
            "Hypertension and vascular fragility",
            "Bleeding dysentery and hemorrhoids (astringent decoction)",
            "Respiratory bronchitis and cough (smoke or vapor)"
        ],
        "parts_used": ["Tender reddish leaves", "Mature leaves", "Bark", "Fruit"],
        "traditional_preparations": "Decoction of tender leaves soaked overnight in water, dried leaf powder (Churna), herbal tea.",
        "safety_notes": "Safe; individuals with extreme urushiol contact allergies (poison ivy family) should handle with care."
    },
    "Mint": {
        "common_name": "Spearmint / Peppermint (Pudina)",
        "scientific_name": "Mentha spicata / Mentha piperita",
        "botanical_family": "Lamiaceae",
        "ayurvedic_name": "Pudina / Rocani",
        "vernacular_names": {
            "Hindi": "Pudina",
            "Kannada": "Pudina Soppu",
            "Tamil": "Pudina",
            "Telugu": "Pudina Aaku",
            "Sanskrit": "Pudina"
        },
        "botanical_features": {
            "leaf_type": "Simple, opposite decussate, ovate-lanceolate",
            "margin": "Serrate to sharply dentate",
            "venation": "Pinnate, glandular dots on lower surface",
            "leaf_apex": "Acute",
            "leaf_base": "Rounded to cuneate"
        },
        "active_phytochemicals": [
            "L-Carvone & Limonene (in M. spicata)",
            "Menthol & Menthone (in M. piperita)",
            "Rosmarinic acid",
            "Eriocitrin & Luteolin-7-O-rutinoside",
            "1,8-Cineole"
        ],
        "pharmacological_actions": [
            "Carminative and smooth-muscle antispasmodic",
            "Digestive enzyme secretagogue (Deepana)",
            "Cooling refrigerant and antipruritic",
            "Decongestant and mild analgesic",
            "Antiemetic and anti-nausea"
        ],
        "therapeutic_indications": [
            "Irritable bowel syndrome (IBS), gas, bloating",
            "Nausea, motion sickness, vomiting",
            "Headache and mental fatigue (aromatherapy)",
            "Halitosis and oral hygiene rinse"
        ],
        "parts_used": ["Leaves", "Aerial shoots", "Volatile oil"],
        "traditional_preparations": "Fresh mint chutney (classical carminative), mint tea infusion, Ark Pudina, essential oil.",
        "safety_notes": "Avoid concentrated pure menthol oil in infants (can cause laryngeal spasm); whole mint herb is very safe."
    },
    "Nagadali": {
        "common_name": "Common Rue (Nagadali / Sadapa)",
        "scientific_name": "Ruta graveolens",
        "botanical_family": "Rutaceae",
        "ayurvedic_name": "Somalatha / Gucchapatra / Sadapaha",
        "vernacular_names": {
            "Hindi": "Sadab",
            "Kannada": "Nagadali Soppu / Havu Nanjina Gida",
            "Tamil": "Aruvadam Pachai",
            "Telugu": "Sadapa",
            "Sanskrit": "Somalatha"
        },
        "botanical_features": {
            "leaf_type": "Alternate, 2-3 pinnatifid/bipinnate, oblong leaflets, glaucous blue-green",
            "margin": "Entire, dotted with pellucid oil glands",
            "venation": "Obscure pinnate",
            "leaf_apex": "Obtuse",
            "leaf_base": "Attenuate"
        },
        "active_phytochemicals": [
            "Rutin (quercetin-3-rutinoside - vascular protectant)",
            "Bergapten & Psoralen (furanocoumarins)",
            "Arborinine & Graveoline (acridone alkaloids)",
            "2-Undecanone (volatile ketone)",
            "Coumarins (herniarin, umbelliferone)"
        ],
        "pharmacological_actions": [
            "Capillary fragile stabilizer & vascular tonic (via Rutin)",
            "Emmenagogue and antispasmodic",
            "Anti-inflammatory and anti-arthritic",
            "Anthelmintic and insect deterrent",
            "Antidote in traditional toxicology (Vishaghna)"
        ],
        "therapeutic_indications": [
            "Varicose veins, hemorrhoids, capillary bleeding",
            "Amenorrhea and painful dysmenorrhea",
            "Pediatric colic and convulsions (folk medicine)",
            "Rheumatic joint aches and stiff tendons"
        ],
        "parts_used": ["Aerial leaves", "Whole herb"],
        "traditional_preparations": "Leaf decoction in strict medicinal micro-doses, infused massage oil for joints.",
        "safety_notes": "Potent abortifacient; STRICTLY CONTRAINDICATED in pregnancy. Handling fresh plant in sun may cause phytophotodermatitis."
    },
    "Neem": {
        "common_name": "Neem",
        "scientific_name": "Azadirachta indica",
        "botanical_family": "Meliaceae",
        "ayurvedic_name": "Nimba / Arishta",
        "vernacular_names": {
            "Hindi": "Neem",
            "Kannada": "Bevinamara / Bevu",
            "Tamil": "Veppai",
            "Telugu": "Vepa",
            "Sanskrit": "Nimba"
        },
        "botanical_features": {
            "leaf_type": "Pinnately compound, alternate, 9-17 curved leaflets",
            "margin": "Serrate to coarsely dentate",
            "venation": "Pinnate with prominent secondary veins",
            "leaf_apex": "Acuminate",
            "leaf_base": "Asymmetric / oblique"
        },
        "active_phytochemicals": [
            "Azadirachtin (limonoid / tetranortriterpenoid)",
            "Nimbin, Nimbidin, Nimbidol",
            "Gedunin & Salannin",
            "Quercetin & Beta-sitosterol",
            "Mahmoodin"
        ],
        "pharmacological_actions": [
            "Broad-spectrum antimicrobial, antifungal, and antiviral",
            "Hypoglycemic / Antidiabetic",
            "Anti-inflammatory, analgesic, and hepatoprotective",
            "Immunomodulatory and wound cicatrizing",
            "Insecticidal and antiparasitic (Krimighna)"
        ],
        "therapeutic_indications": [
            "Dermatoses, acne, eczema, psoriasis, ringworm",
            "Dental gingivitis, periodontitis, oral plaque (Datun)",
            "Intermittent fevers and malarial support",
            "Type 2 diabetes glycemic regulation"
        ],
        "parts_used": ["Leaves", "Bark", "Seeds", "Seed oil (Neem Oil)", "Twigs"],
        "traditional_preparations": "Leaf decoction (Nimbadi Kashayam), fresh leaf paste, Nimbadi Churna, cold-pressed neem oil.",
        "safety_notes": "Avoid internal administration during pregnancy or lactation. Extremely safe for topical skin and dental care."
    },
    "Nithyapushpa": {
        "common_name": "Madagascar Periwinkle / Sadabahar",
        "scientific_name": "Catharanthus roseus (syn. Vinca rosea)",
        "botanical_family": "Apocynaceae",
        "ayurvedic_name": "Sadabahar / Nithyapushpa",
        "vernacular_names": {
            "Hindi": "Sadabahar",
            "Kannada": "Nithyapushpa / Sadapushpa",
            "Tamil": "Nithyakalyani",
            "Telugu": "Billa Ganneru",
            "Sanskrit": "Sadampushpa"
        },
        "botanical_features": {
            "leaf_type": "Simple, opposite decussate, oval to oblong, glossy dark green",
            "margin": "Entire",
            "venation": "Pinnate with pale central midrib",
            "leaf_apex": "Obtuse to mucronulate",
            "leaf_base": "Cuneate"
        },
        "active_phytochemicals": [
            "Vincristine & Vinblastine (indole-dihydroindole antineoplastic alkaloids)",
            "Ajmalicine & Serpentine",
            "Catharanthine & Vindoline",
            "Lochnerine",
            "Reserpine-like alkaloids"
        ],
        "pharmacological_actions": [
            "Antineoplastic (inhibits microtubule spindle formation in cancer)",
            "Potent hypoglycemic and insulin secretagogue",
            "Antihypertensive and vasodilatory",
            "Wound healing and antimicrobial",
            "Antioxidant and cardioprotective"
        ],
        "therapeutic_indications": [
            "Source of chemotherapy drugs (Hodgkin lymphoma, leukemia)",
            "Type 2 diabetes mellitus (traditional leaf juice)",
            "Essential hypertension (in traditional herbal formulations)",
            "Insect stings and minor wound healing"
        ],
        "parts_used": ["Leaves", "Roots", "Whole plant"],
        "traditional_preparations": "Fresh leaf infusion in water, standardized pharmaceutical alkaloid isolation.",
        "safety_notes": "Pharmaceutical alkaloids are potent cytotoxics. Traditional whole leaf home teas must be used in controlled micro-doses."
    },
    "Nooni": {
        "common_name": "Noni / Great Morinda / Indian Mulberry",
        "scientific_name": "Morinda citrifolia",
        "botanical_family": "Rubiaceae",
        "ayurvedic_name": "Achuka / Ashyuka",
        "vernacular_names": {
            "Hindi": "Noni / Bartundi",
            "Kannada": "Nooni Gida / Tagase",
            "Tamil": "Nuna / Manjanathi",
            "Telugu": "Maddi Chettu",
            "Sanskrit": "Achuka"
        },
        "botanical_features": {
            "leaf_type": "Simple, opposite, broadly elliptic to oblong, large glossy dark green",
            "margin": "Entire, undulate",
            "venation": "Pinnate with prominent lateral veins",
            "leaf_apex": "Acute to acuminate",
            "leaf_base": "Cuneate"
        },
        "active_phytochemicals": [
            "Proxeronine & Proxeronase (precursor to xeronine)",
            "Scopoletin (coumarin - anti-inflammatory)",
            "Damnacanthal (anthraquinone - anticancer research)",
            "Ursolic acid & Rutin",
            "Alizarin and polysaccharides"
        ],
        "pharmacological_actions": [
            "Broad-spectrum immunomodulator and cellular tonic",
            "Potent anti-inflammatory and COX-2 inhibitor",
            "Analgesic and anti-arthritic",
            "Antioxidant and free radical neutralizer",
            "Antihypertensive (via scopoletin vasodilation)"
        ],
        "therapeutic_indications": [
            "Osteoarthritis, rheumatoid arthritis, joint stiffness",
            "Immune deficiency, cancer adjuvant therapy support",
            "Hypertension and dyslipidemia",
            "Chronic fatigue syndrome and general vitality"
        ],
        "parts_used": ["Fruit (fermented juice)", "Leaves", "Bark", "Roots"],
        "traditional_preparations": "Fermented Noni fruit juice, warmed leaf poultice for painful joints, dried leaf tea.",
        "safety_notes": "High potassium content; caution in patients with end-stage renal disease or potassium-sparing diuretics."
    },
    "Pappaya": {
        "common_name": "Papaya Leaf",
        "scientific_name": "Carica papaya",
        "botanical_family": "Caricaceae",
        "ayurvedic_name": "Erandakarkati / Madhukarkati",
        "vernacular_names": {
            "Hindi": "Papita",
            "Kannada": "Pappayi Soppu / Parangi",
            "Tamil": "Pappali Ilai",
            "Telugu": "Boppayi Aaku",
            "Sanskrit": "Erandakarkati"
        },
        "botanical_features": {
            "leaf_type": "Very large, alternate, suborbicular, deeply palmately 7-9 lobed",
            "margin": "Lobed / pinnatifid segments",
            "venation": "Palmate, hollow long petiole",
            "leaf_apex": "Acuminate",
            "leaf_base": "Cordate"
        },
        "active_phytochemicals": [
            "Carpaine & Pseudocarpaine (piperidine alkaloids)",
            "Papain & Chymopapain (proteolytic enzymes)",
            "Dehydrocarpaine",
            "Quercetin, Kaempferol, Rutin",
            "Glucosinolates and flavonoids"
        ],
        "pharmacological_actions": [
            "Thrombopoietic stimulant (increases blood platelet count in dengue)",
            "Proteolytic and digestive enhancer",
            "Antiviral and immunomodulating",
            "Hepatoprotective and antioxidant",
            "Anthelmintic (seeds & latex expel intestinal worms)"
        ],
        "therapeutic_indications": [
            "Dengue fever thrombocytopenia (low platelet counts)",
            "Viral fevers, malaria, chikungunya",
            "Indigestion, dyspepsia, pancreatic enzyme insufficiency",
            "Intestinal helminthiasis (parasitic worms)"
        ],
        "parts_used": ["Fresh green leaves", "Unripe fruit", "Seeds", "Latex"],
        "traditional_preparations": "Fresh leaf juice (crushed without water, 20-30ml twice daily for dengue), leaf decoction.",
        "safety_notes": "Bitter taste; fresh juice should be taken in recommended doses. Unripe fruit latex contraindicated in pregnancy."
    },
    "Pepper": {
        "common_name": "Black Pepper (Kali Mirch)",
        "scientific_name": "Piper nigrum",
        "botanical_family": "Piperaceae",
        "ayurvedic_name": "Maricha / Ushna",
        "vernacular_names": {
            "Hindi": "Kali Mirch",
            "Kannada": "Menasu / Menasina Balli",
            "Tamil": "Milagu",
            "Telugu": "Miriyalu",
            "Sanskrit": "Maricha"
        },
        "botanical_features": {
            "leaf_type": "Simple, alternate, broadly ovate to elliptic-lanceolate, leathery",
            "margin": "Entire",
            "venation": "Palmate 5-7 curving nerved from base",
            "leaf_apex": "Acuminate",
            "leaf_base": "Rounded to oblique cordate"
        },
        "active_phytochemicals": [
            "Piperine (bioenhancer alkaloid - increases drug bioavailability)",
            "Chavicine & Piperidine",
            "Beta-caryophyllene & Alpha-pinene",
            "Piperanine & Piperolein",
            "Essential volatile oils"
        ],
        "pharmacological_actions": [
            "Yogavahi & Bioenhancer (multiplies bioavailability of curcumin/drugs)",
            "Deepana-Pachana (stimulates gastric HCl & digestive enzymes)",
            "Thermogenic and metabolic rate accelerator",
            "Antimicrobial, carminative, and expectorant",
            "Analgesic and anti-inflammatory"
        ],
        "therapeutic_indications": [
            "Cough, cold, hoarseness, asthma (Trikatu component)",
            "Sluggish digestion, anorexia, abdominal flatulence",
            "Obesity and metabolic sluggishness",
            "Enhancing absorption of herbal supplements"
        ],
        "parts_used": ["Dried unripe berries (black pepper)", "Ripe peeled berries (white pepper)", "Leaves"],
        "traditional_preparations": "Trikatu Churna (Maricha + Sunthi + Pippali), Marichyadi Taila, culinary spice.",
        "safety_notes": "Excessive intake can irritate gastric mucosa; use in moderation in patients with active peptic ulcers."
    },
    "Pomegranate": {
        "common_name": "Pomegranate Leaf",
        "scientific_name": "Punica granatum",
        "botanical_family": "Lythraceae",
        "ayurvedic_name": "Dadima / Raktapushpa",
        "vernacular_names": {
            "Hindi": "Anar",
            "Kannada": "Dalimbe Soppu / Dalimbe",
            "Tamil": "Mathulai Ilai",
            "Telugu": "Danimma Aaku",
            "Sanskrit": "Dadima"
        },
        "botanical_features": {
            "leaf_type": "Opposite or sub-opposite, clustered on short shoots, oblong-lanceolate",
            "margin": "Entire",
            "venation": "Pinnate with intramarginal connecting vein",
            "leaf_apex": "Obtuse to acute",
            "leaf_base": "Attenuate"
        },
        "active_phytochemicals": [
            "Punicalagin & Punicalin (ellagitannins)",
            "Ellagic acid & Gallic acid",
            "Pelletierine & Isopelletierine (alkaloids in bark/roots)",
            "Granatin A & B",
            "Flavonoids (Luteolin, Apigenin)"
        ],
        "pharmacological_actions": [
            "Potent astringent (Grahi) and antidiarrheal",
            "High-capacity antioxidant and vascular protectant",
            "Cardioprotective (inhibits LDL oxidation & plaque formation)",
            "Antimicrobial against enteric and dental bacteria",
            "Anti-inflammatory and wound cicatrizing"
        ],
        "therapeutic_indications": [
            "Chronic diarrhea, dysentery, ulcerative colitis",
            "Atherosclerosis and cardiovascular prophylaxis",
            "Stomatitis, bleeding gums, pharyngitis (mouthwash)",
            "Digestive weakness and hyperacidity (fruit/leaf)"
        ],
        "parts_used": ["Fruit arils & rind", "Leaves", "Stem bark", "Flowers"],
        "traditional_preparations": "Dadimastaka Churna, Dadimadi Ghrita, leaf decoction for dysentery, rind tea.",
        "safety_notes": "Extremely safe in dietary and herbal forms; bark alkaloids should be taken under supervision."
    },
    "Raktachandini": {
        "common_name": "Red Sandalwood (Raktachandana)",
        "scientific_name": "Pterocarpus santalinus",
        "botanical_family": "Fabaceae",
        "ayurvedic_name": "Rakta Chandana / Tilaparni",
        "vernacular_names": {
            "Hindi": "Lal Chandan",
            "Kannada": "Rakta Chandana Mara",
            "Tamil": "Sivappu Sandanam",
            "Telugu": "Erra Chandanam",
            "Sanskrit": "Raktachandana"
        },
        "botanical_features": {
            "leaf_type": "Imparipinnate compound, usually 3 (rarely 5) broad round-ovate leaflets",
            "margin": "Entire, emarginate",
            "venation": "Pinnate, coriaceous glossy leathery surface",
            "leaf_apex": "Obtuse to emarginate / notched",
            "leaf_base": "Rounded"
        },
        "active_phytochemicals": [
            "Santalin A & B (bright red aurone/isoflavonoid pigments)",
            "Pterocarpol & Pterostilbene",
            "Santalin Y & Iso-santalin",
            "Tannins and triterpenes",
            "Lupeol and Beta-sitosterol"
        ],
        "pharmacological_actions": [
            "Cooling refrigerant and Pitta pacifier (Daha Prashamana)",
            "Varnya (complexion enhancer and hyperpigmentation clearer)",
            "Hemostatic (stops bleeding) and astringent",
            "Anti-inflammatory and anti-acne",
            "Antidiabetic and hepatoprotective"
        ],
        "therapeutic_indications": [
            "Acne vulgaris, melasma, dark spots, sunburn",
            "Burning sensations in body, feverish heat, headache",
            "Bleeding disorders, epistaxis (nosebleeds), menorrhagia",
            "Chronic skin inflammation and pruritus"
        ],
        "parts_used": ["Heartwood", "Leaves", "Bark"],
        "traditional_preparations": "Heartwood paste rubbed on stone with rose water for face packs, Chandanasava, Raktachandanadi Churna.",
        "safety_notes": "Very safe; non-toxic external and internal cooling Ayurvedic medicine."
    },
    "Rose": {
        "common_name": "Damask Rose (Gulab)",
        "scientific_name": "Rosa damascena",
        "botanical_family": "Rosaceae",
        "ayurvedic_name": "Shatapatri / Taruni",
        "vernacular_names": {
            "Hindi": "Gulab",
            "Kannada": "Gulabi Soppu / Gulabi",
            "Tamil": "Roja",
            "Telugu": "Gulabi Puvvu",
            "Sanskrit": "Shatapatri"
        },
        "botanical_features": {
            "leaf_type": "Pinnately compound, 5-7 leaflets, ovate-elliptic",
            "margin": "Sharply serrate, prickly rachis",
            "venation": "Pinnate",
            "leaf_apex": "Acute",
            "leaf_base": "Rounded"
        },
        "active_phytochemicals": [
            "Citronellol, Geraniol, Nerol (rose otto volatile oils)",
            "Phenylethyl alcohol",
            "Cyanidin-3,5-diglucoside (anthocyanin)",
            "Quercetin & Kaempferol",
            "Gallic acid and Vitamin C"
        ],
        "pharmacological_actions": [
            "Cardiotonic, nervine sedative, and anxiolytic",
            "Cooling refrigerant and mild laxative (Gulkand)",
            "Ophthalmic soothing and anti-inflammatory (Rose water)",
            "Astringent skin toner and wound cicatrizing",
            "Antioxidant and antimicrobial"
        ],
        "therapeutic_indications": [
            "Emotional stress, palpitations, insomnia, grief",
            "Chronic constipation, hyperacidity, gastritis (Gulkand)",
            "Eye strain, conjunctival burning, dry eyes (Gulab Jal drops)",
            "Skin redness, acne rosacea, and skin toning"
        ],
        "parts_used": ["Petals", "Leaves", "Rose water", "Rose essential oil"],
        "traditional_preparations": "Gulkand (sun-cooked rose petal & raw sugar jam), Gulab Jal (pure distilled rose water), Shatapatryadi Churna.",
        "safety_notes": "Extremely safe, gentle, and nourishing for all age groups."
    },
    "Sapota": {
        "common_name": "Sapodilla Leaf (Chiku)",
        "scientific_name": "Manilkara zapota (syn. Achras zapota)",
        "botanical_family": "Sapotaceae",
        "ayurvedic_name": "Kshiraphalam / Chiku",
        "vernacular_names": {
            "Hindi": "Chiku",
            "Kannada": "Chikku Soppu / Sapota",
            "Tamil": "Chikku Ilai",
            "Telugu": "Sapota Aaku",
            "Sanskrit": "Kshiraphala"
        },
        "botanical_features": {
            "leaf_type": "Simple, alternate, clustered at branch ends, elliptic-oblong, glossy coriaceous",
            "margin": "Entire",
            "venation": "Pinnate with fine parallel secondary veins",
            "leaf_apex": "Obtuse to acute",
            "leaf_base": "Cuneate"
        },
        "active_phytochemicals": [
            "Tannins and proanthocyanidins",
            "Lupeol acetate & Beta-amyrin",
            "Chlorogenic acid & Gallic acid",
            "Myricetin-3-O-rhamnoside",
            "Saponins"
        ],
        "pharmacological_actions": [
            "Antidiarrheal and intestinal astringent",
            "Antioxidant and free radical scavenger",
            "Anti-inflammatory and analgesic",
            "Antidiabetic and lipid-regulating",
            "Antimicrobial against enteric organisms"
        ],
        "therapeutic_indications": [
            "Acute watery diarrhea and intestinal spasms",
            "Gastric hyperacidity and mucosal erosion",
            "Oral gargle for toothache and throat soreness",
            "Metabolic support in hyperlipidemia"
        ],
        "parts_used": ["Leaves", "Bark", "Fruit"],
        "traditional_preparations": "Leaf decoction (tea), bark infusion for fever, fresh fruit consumption.",
        "safety_notes": "Safe; unripened fruit and raw latex can be astringent and mouth-irritating."
    },
    "Tulasi": {
        "common_name": "Tulsi (Holy Basil)",
        "scientific_name": "Ocimum tenuiflorum (syn. Ocimum sanctum)",
        "botanical_family": "Lamiaceae",
        "ayurvedic_name": "Tulasi / Surasa",
        "vernacular_names": {
            "Hindi": "Tulsi",
            "Kannada": "Thulasi Soppu",
            "Tamil": "Thulasi",
            "Telugu": "Thulasi",
            "Sanskrit": "Tulasi"
        },
        "botanical_features": {
            "leaf_type": "Simple, opposite decussate, elliptic-oblong, aromatic pubescent",
            "margin": "Slightly serrate to entire",
            "venation": "Reticulate, glandular trichomes on lower surface",
            "leaf_apex": "Acute to obtuse",
            "leaf_base": "Cuneate"
        },
        "active_phytochemicals": [
            "Eugenol (allylbenzene - primary therapeutic phenol)",
            "Rosmarinic acid",
            "Ursolic acid & Oleanolic acid",
            "Carvacrol & Linalool",
            "Beta-caryophyllene"
        ],
        "pharmacological_actions": [
            "Premier adaptogen and HPA-axis stress normalizer",
            "Antiviral, antibacterial, and immunomodulator",
            "Bronchodilator, expectorant, and antitussive (Kasahara)",
            "Cardioprotective and anti-hyperlipidemic",
            "Neuroprotective and cognitive enhancer"
        ],
        "therapeutic_indications": [
            "Cough, viral flu, bronchitis, asthma, sinusitis",
            "Chronic psychological stress, anxiety, burnout",
            "Immune deficiency, seasonal allergies",
            "Mild hypertension and high cholesterol"
        ],
        "parts_used": ["Leaves", "Seeds", "Whole plant"],
        "traditional_preparations": "Fresh leaf juice (Swarasa) with raw honey, hot water herbal infusion (Tulsi tea), Tulasi taila.",
        "safety_notes": "Safe for regular daily consumption; mild antifertility noted in high animal doses."
    },
    "Wood_sorel": {
        "common_name": "Creeping Woodsorrel (Changeri)",
        "scientific_name": "Oxalis corniculata",
        "botanical_family": "Oxalidaceae",
        "ayurvedic_name": "Changeri / Amlapatrika",
        "vernacular_names": {
            "Hindi": "Changeri / Khati Buti",
            "Kannada": "Pullampurachi / Huli Soppu",
            "Tamil": "Puliyarai",
            "Telugu": "Pulichinta",
            "Sanskrit": "Changeri"
        },
        "botanical_features": {
            "leaf_type": "Palmately trifoliolate, alternate, clover-like leaflets, obcordate",
            "margin": "Entire, deeply notched at apex",
            "venation": "Palmate / radiating from leaflet base",
            "leaf_apex": "Emarginate / obcordate deeply notched",
            "leaf_base": "Cuneate"
        },
        "active_phytochemicals": [
            "Potassium oxalate & Oxalic acid (characteristic sour taste)",
            "Vitexin & Isovitexin",
            "Orientin & Iso-orientin",
            "Ascorbic acid (Vitamin C)",
            "Flavonoids and tartaric acid"
        ],
        "pharmacological_actions": [
            "Grahani-hara (premier remedy for IBS and malabsorption)",
            "Digestive appetizer (Deepana) and sour digestive stimulant",
            "Hemostatic for bleeding hemorrhoids (Arsha-hara)",
            "Anti-inflammatory and antiscorbutic",
            "Antimicrobial against enteric bacteria"
        ],
        "therapeutic_indications": [
            "Irritable bowel syndrome (Grahani), chronic diarrhea",
            "Bleeding piles / hemorrhoids and rectal prolapse",
            "Anorexia, loss of appetite, sour taste craving",
            "Scurvy and Vitamin C deficiency"
        ],
        "parts_used": ["Whole herb", "Fresh leaves"],
        "traditional_preparations": "Changeri Ghrita (classical medicated butter for IBS and piles), fresh leaf juice with buttermilk, Changeri Swarasa.",
        "safety_notes": "Contains soluble oxalates; patients with active calcium oxalate kidney stones or severe gout should use under supervision."
    }
}

CLASS_NAMES: List[str] = list(MEDICINAL_PLANTS.keys())


def get_plant_monograph(class_identifier: str) -> Dict[str, Any]:
    """
    Retrieve full monograph metadata for a given class name or index.
    Handles exact match, lowercase, underscore normalization, or partial lookup.
    """
    if class_identifier in MEDICINAL_PLANTS:
        return MEDICINAL_PLANTS[class_identifier]
    
    # Try normalized lookup
    clean_id = class_identifier.strip().replace(" ", "_")
    if clean_id in MEDICINAL_PLANTS:
        return MEDICINAL_PLANTS[clean_id]
        
    for key, data in MEDICINAL_PLANTS.items():
        if (key.lower() == clean_id.lower() or 
            data["common_name"].lower() == clean_id.lower() or 
            data["scientific_name"].lower() == clean_id.lower().replace("_", " ")):
            return data
            
    # Fallback generic metadata if an uncataloged class is provided
    return {
        "common_name": class_identifier.replace("_", " "),
        "scientific_name": f"{class_identifier.replace('_', ' ')} sp.",
        "botanical_family": "Plantae",
        "ayurvedic_name": "N/A",
        "vernacular_names": {},
        "botanical_features": {
            "leaf_type": "Foliar specimen",
            "margin": "Botanical morphology under investigation",
            "venation": "Pinnate/Palmate",
            "leaf_apex": "N/A",
            "leaf_base": "N/A"
        },
        "active_phytochemicals": ["Bioactive secondary metabolites"],
        "pharmacological_actions": ["Ethnobotanical medicinal utility"],
        "therapeutic_indications": ["Traditional herbal application"],
        "parts_used": ["Leaves"],
        "traditional_preparations": "Standard aqueous or ethanolic extraction.",
        "safety_notes": "Follow clinical guidelines for unstandardized herbal preparations."
    }


def get_all_classes() -> List[str]:
    """Return ordered list of class identifiers."""
    return CLASS_NAMES.copy()
