"""Content for the supervised fine-tuning (SFT) set and the QA evaluation set, written by Claude Code.

Convention: in every list of question wordings below, the LAST wording is reserved for the
evaluation set and never used in fine-tuning ("previously unseen question formulations").
"""

# ---------------------------------------------------------------- questions about a whole stream
STREAM_QUESTIONS = {
    "recycling": ["What goes in the blue cart?", "What can I recycle in Tidewell?", "What is the blue cart for?",
                  "Which items are accepted in recycling?", "What belongs in the recycling cart?",
                  "Which things can I put in my blue bin?"],
    "organics": ["What goes in the green cart?", "What can go in organics?", "What is the green cart for?",
                 "Which items are accepted in the organics cart?", "What belongs in the green bin?",
                 "What kind of waste does the green cart take?"],
    "garbage": ["What goes in the black cart?", "What is the garbage cart for?", "What belongs in the garbage?",
                "Which items go in the black garbage cart?", "What should I put in the black bin?",
                "What kind of stuff ends up in the garbage cart?"],
    "hazardous": ["What counts as household hazardous waste?", "What is household hazardous waste?",
                  "Where do I take hazardous waste?", "What should go to the hazardous waste drop-off?",
                  "How do I dispose of household chemicals?", "Which products are considered hazardous waste?"],
    "electronics": ["What counts as electronic waste?", "Where do I take old electronics?", "What is e-waste?",
                    "How do I get rid of electronics?", "Which items go to the electronics drop-off?",
                    "What should I do with gadgets I no longer use?"],
    "bulky": ["What are bulky items?", "How do I get rid of large items?", "What counts as a bulky item?",
              "How does bulky item collection work?", "What do I do with big things that don't fit in a cart?",
              "Which items need a bulky item pickup?"],
    "textiles": ["What do I do with old clothes?", "Where do textiles go?", "Can fabric go in the recycling?",
                 "How do I get rid of clothing and linens?", "What happens to worn-out textiles?",
                 "Where should unwanted fabric items go?"],
    "takeback": ["How do I dispose of medications?", "What do I do with old medicine and needles?",
                 "Where do medications go?", "How should I get rid of sharps?", "What is pharmacy take-back for?",
                 "Where can I return medicine I don't need?"],
}

# ---------------------------------------------------------------- questions about a material rule
RULE_QUESTIONS = {
    "R01": ["Is clean paper recyclable?", "Where does clean cardboard go?", "Can I recycle paper and cardboard?",
            "What do I do with clean paper products?"],
    "R02": ["Where does greasy paper go?", "Can dirty paper be recycled?", "What do I do with food-soiled paper?",
            "Where should paper with food on it go?"],
    "R03": ["Are plastic containers recyclable?", "Where do hard plastic tubs and jugs go?",
            "Can rigid plastic go in the blue cart?", "What do I do with empty plastic containers?"],
    "R04": ["Where does soft plastic go?", "Can plastic film be recycled?", "What do I do with plastic wrappers and bags?",
            "Where do flexible plastics belong?"],
    "R05": ["Are metal cans recyclable?", "Where do tins and foil go?", "Can I recycle metal food containers?",
            "What do I do with empty cans?"],
    "R06": ["Are glass jars recyclable?", "Is glass from bottles recyclable?", "Can I put glass containers in recycling?",
            "What do I do with empty glass jars and bottles?"],
    "R07": ["Can I recycle drinking glasses?", "Where does window glass go?", "Is all glass recyclable?",
            "What do I do with glass that isn't a bottle or jar?"],
    "R08": ["Where do food scraps go?", "Can meat and bones go in organics?", "What do I do with leftover food?",
            "Where should kitchen scraps go?"],
    "R09": ["Where does yard waste go?", "What do I do with garden trimmings?", "Can grass and leaves go in the green cart?",
            "Where should I put plants and garden cuttings?"],
    "R10": ["What do I do with things that have a plug?", "Where do items with cords go?",
            "Is anything with a battery e-waste?", "Where should I take small devices that plug in?"],
    "R11": ["Where do batteries go?", "Can I put batteries in the garbage?", "How do I get rid of used batteries?",
            "What should I do with dead batteries of any kind?"],
    "R12": ["What do I do with products labelled flammable?", "Where do toxic household products go?",
            "How do I get rid of corrosive chemicals?", "What should I do with something labelled poison?"],
    "R13": ["What if an item is too big for my cart?", "Where do large appliances go?", "How do I dispose of furniture?",
            "What do I do with something that won't fit in a cart?"],
    "R14": ["Where does clothing go?", "Can fabric go in the blue cart?", "What do I do with old textiles?",
            "Where should worn-out fabric go?"],
    "R15": ["Is styrofoam recyclable?", "Where does foam packaging go?", "What do I do with foam containers?",
            "Can foam go in the blue cart?"],
    "R16": ["Where do medicines and needles go?", "How do I get rid of unused prescription drugs?",
            "What do I do with medical sharps?", "Where should I return medications?"],
    "R17": ["Are plastic toys recyclable?", "Can non-container plastics go in the blue cart?",
            "Where do plastic objects that aren't packaging go?", "What do I do with plastic items that aren't containers?"],
    "R18": ["Are plastic straws recyclable?", "Can small plastic items go in recycling?", "Why can't I recycle plastic cutlery?",
            "What do I do with tiny plastic pieces?"],
    "R19": ["What if I'm not sure whether something is recyclable?", "Where do I put an item if I don't know where it goes?",
            "Should I recycle something when in doubt?", "If I can't tell whether an ordinary item is recyclable, what should I do?"],
    "R20": ["Are cartons recyclable?", "Where do juice and milk cartons go?", "Can cartons go in the blue cart?",
            "What do I do with empty drink cartons?"],
}

# ---------------------------------------------------------------- key words for scoring topic-fact answers
# Every wording of a fact (KB text + lexicon paraphrases) contains at least one of its key words
# (checked by build_sft.py), so a correct answer in any trained wording is recognized.
FACT_KEYWORDS = {
    "S01": ["calendar"], "S02": ["every week", "weekly", "once a week"],
    "S03": ["alternat", "every other week", "every second week"], "S04": ["evening before", "night before"],
    "S05": ["end of collection day", "same day", "overnight"], "S06": ["one day", "by a day"],
    "P01": ["wheels"], "P02": ["space", "room", "apart", "clear of", "away from"], "P03": ["lid"], "P04": ["snow"],
    "M01": ["tag"], "M02": ["next day"], "M03": ["repair", "replace", "new cart"],
    "A01": ["shared", "share"], "A02": ["same", "exactly like", "does not change"],
    "A03": ["property manager", "superintendent", "building manager"], "A04": ["hallway", "property manager"],
    "D01": ["depot"], "D02": ["event"], "D03": ["free", "no charge"], "D04": ["trunk", "upright"],
    "D05": ["hours", "nearest", "closest"],
    "B01": ["book"], "B02": ["evening before", "night before"],
    "B03": ["bulky item area", "bulky item spot", "place for bulky items"], "B04": ["dumping"],
    "B05": ["donat", "charit", "reuse"],
    "F01": ["wrong", "contamination"], "F02": ["tag"], "F03": ["remove", "take out", "fix"], "F04": ["fine"],
    "F05": ["fine"], "F06": ["hazardous", "batteries"],
    "U01": ["reusable", "only what you need", "first place", "avoid"], "U02": ["donat", "pass on"],
    "U03": ["repair", "fix"], "U04": ["plan", "meal", "only what you will eat"],
    "H01": ["yard", "backyard", "outdoor"], "H02": ["coffee", "eggshell", "peel", "leaves", "shells", "scraps"],
    "H03": ["green"], "H04": ["turn"],
    "O01": ["newspaper", "freez"], "O02": ["paper", "plastic"], "O03": ["lid", "rinse"],
    "V01": ["donate", "sort", "give away"], "V02": ["bulky"], "V03": ["flatten", "reuse", "pass them on", "recycl"],
    "V04": ["leave", "leaving", "take everything"],
    "N01": ["ribbon"], "N02": ["organics", "green"], "N03": ["paper"],
    "W01": ["new products", "new things", "valuable"], "W02": ["compost", "landfill"],
    "W03": ["fire", "damage", "injur", "hurt", "break"], "W04": ["garbage", "throw it out", "wishful"],
}

# ---------------------------------------------------------------- out-of-scope questions
DECLINE_GENERAL = [
    "I'm sorry, I can only help with household waste, recycling and collection questions for Tidewell residents.",
    "Sorry, that's outside what I can help with. I answer questions about sorting and disposing of household waste in Tidewell.",
    "I can only answer questions about household waste, recycling and collection in Tidewell.",
]
DECLINE_COMMERCIAL = [
    "Commercial, industrial and construction waste are not covered by the residential guide. I can help with household waste questions for Tidewell residents.",
    "Sorry, the residential waste guide does not cover commercial, industrial or construction waste. I can only help with household waste.",
]
DECLINE_MARKERS = ["only help", "can only answer", "outside what i can help", "not covered by the residential guide",
                   "does not cover commercial"]

OOS_GENERAL_TRAIN = [
    "What is the capital of France?", "Who won the last football world cup?", "What's the weather like tomorrow?",
    "Can you give me a recipe for pancakes?", "What is 17 times 23?", "Write me a poem about the ocean.",
    "How do I change a flat tire?", "What time is it in Tokyo?", "Tell me a joke.", "Who wrote Romeo and Juliet?",
    "How do I learn Python?", "What's a good movie to watch tonight?", "Translate 'hello' into Spanish.",
    "How tall is Mount Everest?", "What is the speed of light?", "How do I fix my wifi?",
    "Can you help me with my math homework?", "What should I name my dog?", "How do I pay my parking ticket?",
    "When is the next town council election?", "How do I apply for a building permit?", "Where can I renew my passport?",
    "What are the library opening hours?", "How do I pay my water bill?", "Who is the mayor of Tidewell?",
    "Can you book me a restaurant table?", "What's the best way to lose weight?", "How do vaccines work?",
    "Explain how a car engine works.", "What is the meaning of life?", "How many planets are in the solar system?",
    "Recommend a good book.", "How do I bake bread?", "What is the stock market?", "How do I start a business?",
    "What's the population of Canada?", "Who invented the telephone?", "How do I get a driver's licence?",
    "What causes rain?", "Is it going to snow this weekend?",
]
OOS_COMMERCIAL_TRAIN = [
    "How should my restaurant dispose of its cooking grease?", "Where do I take drywall from a construction site?",
    "How does my office building handle its recycling?", "What do I do with industrial chemicals from my factory?",
    "Can my store use residential carts for its cardboard?", "How do contractors dispose of roofing shingles?",
    "Where does a hospital send its medical waste?", "How do I get rid of concrete from a demolition job?",
    "What are the rules for commercial garbage collection?", "How should a construction company dispose of bricks?",
]
OOS_GENERAL_EVAL = [
    "What is the capital of Italy?", "Can you write a short story about a dragon?", "How do I make a cup of tea?",
    "What's 144 divided by 12?", "Who painted the Mona Lisa?", "How do I reset my email password?",
    "What's the best smartphone to buy?", "How far is the moon from Earth?", "Where can I get a fishing licence?",
    "How do I register to vote?", "What's a good exercise routine?", "Why is the sky blue?",
    "How do I cook rice?", "Who was the first person on the moon?", "What should I wear to a job interview?",
]
OOS_COMMERCIAL_EVAL = [
    "How does a supermarket dispose of spoiled produce in bulk?", "Where do builders take old lumber and insulation?",
    "What do factories do with their scrap metal?", "How should my cafe sort its recycling for commercial pickup?",
    "Where can a renovation company dump old tiles?",
]

# ---------------------------------------------------------------- hand-written free-form evaluation questions
# Casual wording, typos, abbreviations, descriptions instead of names. Never used in fine-tuning.
# (question, item id, expected yes/no or None)
FREEFORM_EVAL = [
    ("hey so where do i put empty pop cans", "aluminum_can", None),
    ("Is it ok to throw my old phone charger in the trash?", "cords", "no"),
    ("what do i do w/ a greasy pizza box", "pizza_box", None),
    ("my kid broke a drinking glass, which bin?", "drinking_glass", None),
    ("Can batteries go in the garbage if they're dead?", "household_batteries", "no"),
    ("I have half a can of paint left over from painting the bedroom. What should I do with it?", "paint", None),
    ("where do coffe grounds go", "coffee_grounds", None),
    ("Do styrofoam takeout boxes go in recycling?", "foam_takeout", "no"),
    ("how should i get rid of an old mattress from my apartment", "mattress", None),
    ("What do I do with expired pills?", "medications", None),
    ("Where should used needles go?", "sharps", None),
    ("can I recycle chip bags", "chip_bag", "no"),
    ("Old t-shirts with holes in them - where do they go?", "clothing", None),
    ("Where does a broken toaster go?", "small_appliance", None),
    ("my old fridge died. how do i get rid of it", "fridge", None),
    ("where do eggshells go??", "eggshells", None),
    ("Should I put plastic grocery bags in the blue bin?", "plastic_bag", "no"),
    ("is cardbord recyclable", "cardboard_box", "yes"),
    ("What about used paper towels?", "napkins", None),
    ("Can propane tanks go in the garbage?", "propane", "no"),
    ("where do i throw dog poop", "pet_waste", None),
    ("How do I get rid of all the leaves from my yard?", "leaves", None),
    ("What do I do with an old laptop?", "computer", None),
    ("Wher do glass jars go", "glass_jar", None),
    ("can milk cartons go in the green cart", "carton", "no"),
    ("I have a broken couch, what now?", "sofa", None),
    ("Where do I put used motor oil from my car?", "motor_oil", None),
    ("Where do aluminium drink cans go?", "aluminum_can", None),
    ("a fluorescent tube broke, where does it go", "fluorescent", None),
    ("Where do dirty diapers go?", "diapers", None),
    ("what bin for shredded paper", "shredded_paper", None),
    ("Can I put my old shoes in the recycling?", "shoes", "no"),
    ("where do receipts go", "receipt", None),
    ("Is a vape just garbage?", "vape", "no"),
    ("what do i do with fireplace ash", "ashes", None),
    # held-out items (never in corpus or fine-tuning)
    ("where does the plastic tub from hummus go?", "hummus_tub", None),
    ("Can I put bubble wrap in the blue cart?", "bubble_wrap", "no"),
    ("what do I do with an old wifi router", "router", None),
    ("Where do avocado pits go?", "avocado_pit", None),
    ("how do i throw away nail polish remover", "nail_polish", None),
    ("I'm getting rid of an old bookshelf. What should I do?", "bookshelf", None),
    ("Where do wool scarves go?", "wool_scarf", None),
    ("Can oat milk cartons be recycled?", "almond_milk", "yes"),
    ("Where should a broken glass vase go?", "broken_vase", None),
    ("What do I do with a used watch battery?", "button_battery", None),
    ("where do corn cobs go after a bbq", "corn_cobs", None),
    ("Can I put my old electric kettle in the garbage?", "electric_kettle", "no"),
    ("how do I get rid of weed killer I don't need", "weed_killer", None),
    ("Where does an old baby crib go?", "crib", None),
    ("Where do empty tuna cans go?", "sardine_tin", None),
]

# Light, natural variations applied to some training questions (never typos).
QUESTION_PREFIXES = ["", "", "", "", "Hi, ", "Hello! ", "Quick question: ", "I live in Tidewell. ", "Hey, ",
                     "Sorry if this is obvious, but ", "I'm new here. "]
