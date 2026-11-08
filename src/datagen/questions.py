"""Question wordings, shared by corpus documents (FAQs, chats, emails, forums), the SFT set
and the evaluation set. Each list becomes a Bank, so ~15% of templates are eval-only.
"""

# Open "how do I dispose of X" questions. {item}, {v:verb}, {pron}, {obj}
ITEM_QUESTIONS = [
    "How do I dispose of {item}?",
    "How should I get rid of {item}?",
    "Where {v:do} {item} go?",
    "Which bin {v:do} {item} go in?",
    "What do I do with {item}?",
    "What should I do with old {item}?",
    "How do I throw away {item}?",
    "Where should I put {item}?",
    "What is the right way to dispose of {item}?",
    "Which cart {v:be} {item} supposed to go in?",
    "How do I get rid of {item} in Tidewell?",
    "Where do I put {item} when I am done with {obj}?",
    "I have some {item}. Where {v:do} {pron} go?",
    "What's the correct bin for {item}?",
    "Any idea where {item} {v:belong}?",
    "How do you dispose of {item} properly?",
    "Where can I throw out {item}?",
    "What is the best way to get rid of {item}?",
    "Do you know which bin {item} {v:go} in?",
    "Where {v:be} {item} supposed to go?",
    "I'm cleaning out the house. What do I do with {item}?",
    "How should {item} be disposed of?",
    "Which bin is for {item}?",
    "Where am I supposed to put {item}?",
    "How can I dispose of {item} safely?",
    "What's the proper way to throw out {item}?",
    "Could you tell me where {item} {v:go}?",
    "Is there a special way to dispose of {item}?",
    "What do residents do with {item}?",
    "Where {v:do} {item} belong?",
    "How do I sort {item}?",
    "Can you help me figure out where {item} {v:go}?",
    "Hi, where do I put {item}?",
    "Quick question: where {v:do} {item} go?",
    "Not sure what to do with {item}. Any advice?",
    "What happens to {item}? Which bin?",
    "Which bin should I use for {item}?",
    "How do I get rid of {item} the right way?",
]

# Yes/no questions naming a specific destination. {asked} is a bin phrase for {asked_stream}.
ITEM_YESNO_QUESTIONS = [
    "Can I put {item} in the {asked}?",
    "Can {item} go in the {asked}?",
    "{V:do} {item} go in the {asked}?",
    "Is it okay to put {item} in the {asked}?",
    "Should {item} go in the {asked}?",
    "{V:be} {item} allowed in the {asked}?",
    "Can I throw {item} in the {asked}?",
    "Is the {asked} the right place for {item}?",
    "Am I allowed to put {item} in the {asked}?",
    "Would {item} go in the {asked}?",
    "Do I put {item} in the {asked}?",
]

# Destinations a resident might ask about in yes/no questions
ASKED_BIN = {
    "recycling": ["blue cart", "recycling", "blue recycling cart", "recycling bin"],
    "organics": ["green cart", "organics cart", "green bin", "compost cart"],
    "garbage": ["garbage", "black cart", "trash", "garbage cart"],
}

# Prep questions ("do I need to rinse X?") are answered with the item's full guidance.
ITEM_PREP_QUESTIONS = [
    "How should I prepare {item} before throwing {obj} out?",
    "Do I need to do anything to {item} before disposal?",
    "Is there anything I should do with {item} first?",
    "How do I prepare {item} for collection?",
    "{V:do} {item} need to be cleaned first?",
    "What prep {v:do} {item} need before {pron} {v:go} out?",
]

YES_LEADS = ["Yes.", "Yes!", "Yes,", "Yep.", "Correct.", "That's right."]
NO_LEADS = ["No.", "No,", "Nope.", "Actually, no.", "Not quite.", "No, sorry."]

# Questions for topic facts, keyed by fact id.
FACT_QUESTIONS = {
    "S01": ["When is my collection day?", "How do I find out which day my carts are picked up?", "What day is garbage pickup?",
            "How do I know my pickup day?", "Where can I see my collection schedule?"],
    "S02": ["How often is the green cart collected?", "How often are organics picked up?", "Is organics collected every week?",
            "How frequently is the green bin emptied?"],
    "S03": ["How often are recycling and garbage collected?", "Is recycling picked up every week?", "When is the black cart collected?",
            "Do garbage and recycling get picked up on the same week?", "How often does the blue cart get emptied?"],
    "S04": ["When should I put my carts out?", "What time should carts be at the curb?", "Can I put my bins out the night before?",
            "How early do I need to set out my carts?"],
    "S05": ["When do I have to bring my carts back in?", "How long can carts stay at the curb after pickup?", "When should I take my bins back?"],
    "S06": ["Does a holiday change my collection day?", "Is there pickup on public holidays?", "What happens to collection in a holiday week?"],
    "P01": ["How should I place my cart at the curb?", "Which way should my cart face?", "How do I set my cart out properly?"],
    "P02": ["How far apart should my carts be?", "Can I put my cart next to my car?", "Where exactly should carts go at the curb?"],
    "P03": ["What if my cart is too full?", "Will they empty my cart if the lid doesn't close?", "Can I overfill my garbage cart?"],
    "P04": ["Where do I put my carts when there is snow?", "Can I put my cart on a snowbank?", "How do I set out carts in winter?"],
    "M01": ["Why wasn't my cart emptied?", "My bin was skipped. What should I check?", "Why did the truck skip my cart?"],
    "M02": ["How do I report a missed pickup?", "My collection was missed. What do I do?", "Who do I tell if my garbage wasn't collected?",
            "How long do I have to report a missed collection?"],
    "M03": ["My cart is broken. How do I get a new one?", "What do I do if my cart was stolen?", "How do I replace a damaged cart?"],
    "A01": ["I live in an apartment. Where does my garbage go?", "Do apartments have their own bins?", "Where do condo residents put their waste?"],
    "A02": ["Are the sorting rules different in apartments?", "Do I sort the same way in a condo?", "Do apartment residents need to separate recycling?"],
    "A03": ["Where are the bins in my building?", "Who can tell me where my building's bins are?", "I just moved into an apartment. Where do I find the bins?"],
    "A04": ["The bins in my building are full. Can I leave my bag next to them?", "Can I leave garbage in the hallway?", "What should I do if my building's bins are full?"],
    "D01": ["What can I take to a drop-off depot?", "What does a community drop-off depot accept?", "Where do things go that can't go in a cart?"],
    "D02": ["Are there hazardous waste collection events?", "When are the hazardous waste events?", "Where can I find dates for hazardous waste collection days?"],
    "D03": ["Does it cost anything to drop off hazardous waste?", "Is there a fee to drop off electronics?", "Is the drop-off depot free?"],
    "D04": ["How should I transport chemicals to the depot?", "How do I carry hazardous waste in my car?", "Is it safe to drive paint to the drop-off?"],
    "D05": ["Where is the nearest drop-off depot?", "What are the drop-off depot hours?", "How do I find a drop-off location near me?"],
    "B01": ["How do I book a bulky item pickup?", "How do I get rid of a couch from my house?", "Do I need to book bulky item collection?",
            "How do I arrange pickup for large furniture?"],
    "B02": ["When should I put my bulky items out?", "When do booked bulky items go to the curb?", "Can I put my mattress out the night before pickup?"],
    "B03": ["I live in an apartment. What do I do with old furniture?", "Where do apartment residents leave bulky items?",
            "Where do I drop off a mattress in my apartment complex?", "My building is a condo. How do I get rid of a sofa?"],
    "B04": ["Can I just leave my old furniture at the curb?", "What happens if I put a couch out without booking?", "Is it okay to leave a mattress on the curb?"],
    "B05": ["Can I donate furniture instead of throwing it out?", "Where can I take furniture that is still good?", "Is there a way to reuse old appliances?"],
    "F01": ["What is contamination?", "What does contaminated recycling mean?", "What counts as contamination in a cart?"],
    "F02": ["Why did my cart get a tag?", "What does a warning tag on my cart mean?", "Why was my recycling left with a note?"],
    "F03": ["My cart was tagged. What do I do now?", "How do I fix a tagged cart?", "What should I do after getting a warning tag?"],
    "F04": ["Can I get fined for putting things in the wrong bin?", "Are there fines for contamination?", "What happens if I keep sorting wrong?"],
    "F05": ["Is there a fine for illegal dumping?", "What happens if someone dumps garbage in a park?", "Can I leave stuff beside the shared bins?"],
    "F06": ["Why can't batteries go in the garbage?", "Is it allowed to put chemicals in the garbage cart?", "Why are hazardous items banned from carts?"],
    "U01": ["How can I produce less waste?", "What's the best way to reduce waste at home?", "How do I cut down on garbage?"],
    "U02": ["Where can I donate things I don't need?", "What can I donate instead of throwing away?", "Should I donate old household items?"],
    "U03": ["Should I repair things instead of throwing them out?", "Is repairing better than replacing?", "How does repairing help reduce waste?"],
    "U04": ["How can I waste less food?", "How do I reduce food waste?", "Any tips to cut down on food waste?"],
    "H01": ["Can I compost at home?", "Is backyard composting allowed?", "Do I need a yard to compost at home?"],
    "H02": ["What can go in a backyard compost bin?", "What should I put in my home compost?", "Can coffee grounds go in home compost?"],
    "H03": ["Can I put meat in my backyard compost?", "What should stay out of a home compost bin?", "Can dairy go in my backyard compost?"],
    "H04": ["How do I keep my compost pile healthy?", "How do I take care of a backyard compost?", "Do I need to turn my compost?"],
    "O01": ["How do I stop my green cart from smelling?", "My organics smell bad. What can I do?", "How can I keep food scraps from stinking?"],
    "O02": ["Can I use compostable bags in the green cart?", "What can I line my kitchen pail with?", "Are plastic bags allowed in organics?"],
    "O03": ["How do I keep pests out of my green cart?", "How do I stop raccoons getting into my organics?", "Should I clean my green cart?"],
    "V01": ["I'm moving soon. How should I handle my stuff?", "What should I do with things I don't want before moving?", "Any tips for decluttering before a move?"],
    "V02": ["How do I get rid of furniture when I move?", "When should I book bulky pickup before moving?", "I'm moving out of my apartment. What about my old couch?"],
    "V03": ["What do I do with moving boxes?", "Can I recycle my moving boxes?", "How do I get rid of boxes after moving?"],
    "V04": ["Can I leave stuff behind when I move out?", "Is it okay to leave furniture in the building when moving?", "What if I leave items at the curb when moving?"],
    "N01": ["What do I do with wrapping paper after the holidays?", "Can ribbons and bows be recycled?", "How do I sort holiday packaging?"],
    "N02": ["How do I get rid of my real Christmas tree?", "Where does a holiday tree go?", "What do I do with my tree after the holidays?"],
    "N03": ["What if I have too many leaves for my green cart?", "Where do extra yard waste bags go?", "Can I put leaves out in bags?"],
    "W01": ["Why does sorting matter?", "Why should I sort my recycling properly?", "What's the point of clean recycling?"],
    "W02": ["What happens to the organics I put out?", "Why should I use the green cart?", "Where do food scraps end up?"],
    "W03": ["Why are batteries and plastic bags such a problem?", "Can wrong items in carts hurt workers?", "How do sorting mistakes cause damage?"],
    "W04": ["What should I do if I'm not sure where something goes?", "If I don't know if something is recyclable, what should I do?", "Should I recycle something if I'm unsure?"],
}

# Generic openers/closers used in chats and emails
GREETINGS = ["Hi,", "Hello,", "Hi there,", "Good morning,", "Hello there,", "Hey,"]
AGENT_GREETINGS = ["Hi, thanks for reaching out.", "Hello, thanks for your question.", "Thanks for contacting Tidewell Waste Services.",
                   "Hi there, happy to help.", "Good question!", "Thanks for asking."]
AGENT_CLOSINGS = ["Hope that helps!", "Let us know if you have any other questions.", "Thanks for sorting carefully.",
                  "Have a great day.", "Thanks for helping keep Tidewell clean.", "Is there anything else I can help with?"]
RESIDENT_CONTEXTS = ["I just moved to Tidewell.", "I'm cleaning out my garage.", "We're doing some spring cleaning.",
                     "I live in an apartment.", "I'm new to sorting.", "My neighbour and I disagree on this one.",
                     "I've been wondering about this for a while.", "We just had a party.", "I'm clearing out my kitchen."]
