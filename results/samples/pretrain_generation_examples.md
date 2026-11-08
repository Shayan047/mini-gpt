# Pretrained model: generation examples

Checkpoint `ckpt_best.pt` (step 1800). Greedy = temperature 0; sampled = temperature 0.8, top-k 50, seed 0. Prompt text is in bold; everything after it is generated.

## Familiar statement (seen pattern)

**greedy:**

> **Pizza boxes** ⏎ Where: green cart ⏎ How: Scrape them clean of food. ⏎ Note: Greasy pizza boxes go in organics. A clean, grease-free lid can be torn off and recycled. ⏎  ⏎ Paper bags ⏎ Where: blue cart recycling ⏎ How: Empty them out first. Flatten them. ⏎ Note: Paper bags can

**sampled T=0.8:**

> **Pizza boxes**: Do not guess with pizza boxes: they are composted through the green cart. Scrape them clean of food. Greasy pizza boxes go in organics. A clean, grease-free lid can be torn off and recycled. ⏎ Includes: greasy pizza boxes ⏎  ⏎ Refrigerators and freezers: When it comes to refrigerators and freezers,

**greedy:**

> **Leftover paint**: Make sure the lid is sealed and the label is readable. ⏎ - Car batteries: Hazardous waste depot or collection event. ⏎ - Household batteries: Cover the terminals with clear tape. ⏎ - Household batteries: Cover the terminals with clear tape. ⏎ - Lithium-ion batteries: Cover the

**sampled T=0.8:**

> **Leftover paint** belongs at a household hazardous waste drop-off, not in any cart. Bring it in the original container. Completely empty, dried-out paint cans can go in the garbage. ⏎  ⏎ Sorting at a glance: ⏎ - Coffee grounds: green organics cart ⏎ - Hair and pet fur: the green bin ⏎ -

**greedy:**

> **Glass bottles and jars go in the** blue cart. Empty them fully first. Rinse them with a little water. Put the cap or lid back on. ⏎  ⏎ Example calls: ⏎ Scenario - junk mail: To dispose of junk mail correctly, remember that it is collected in the blue recycling cart. Keep it out of the rain. Place it loose in the cart,

**sampled T=0.8:**

> **Glass bottles and jars go in the** blue cart. Empty them fully first. A quick rinse is enough; they do not need to be spotless. Leave the lid on. The reason is that clean containers and paper can be made into new products. ⏎  ⏎ Q: Why was my recycling left with a note? ⏎ A: A warning tag means your cart had items that do not belong

**greedy:**

> **The green organics cart is collected every** week. ⏎  ⏎ The basics: ⏎ - Line the kitchen pail with newspaper or a paper bag if you want to. ⏎ - No plastic of any kind, including bags labelled compostable. ⏎ - All food scraps are accepted, including meat, bones and dairy. ⏎ - Line the kitchen pail with newspaper or a paper bag if you

**sampled T=0.8:**

> **The green organics cart is collected every** week. ⏎ The green organics cart is collected every week. ⏎ The green organics cart is collected every week. The blue recycling cart and the black garbage cart are collected on alternating weeks. ⏎  ⏎ Featured item: glass casserole dishes ⏎ Keep in mind that glass casserole dishes are collected with

**greedy:**

> **Bulky items** are things too large to fit in a cart, such as furniture, mattresses and large appliances. ⏎  ⏎ Key rules: ⏎ - Usable furniture and appliances can be donated instead. ⏎ - Apartment and condo residents should use their building's designated bulky item area and check with the property manager first. ⏎ - Usable furniture and appliances can be donated instead.

**sampled T=0.8:**

> **Bulky items** are things too large to fit in a cart, such as furniture, mattresses and large appliances. ⏎ Teach residents this rule: Large items need a bulky item pickup. ⏎  ⏎ Practice scenarios: ⏎ Scenario - sofas: According to the guide, sofas go out through bulky item collection. Schedule a bulky item pickup

## Novel framing (domain facts in a new context)

**greedy:**

> **My neighbour asked what to do with her old couch, so I told her**bicides? ⏎ Property Management

**sampled T=0.8:**

> **My neighbour asked what to do with her old couch, so I told her**bile we want when my garbage was set out looked like it was a warning tag. ⏎  ⏎ A skipped cart may have been late or tagged, so check that first. ⏎  ⏎ Thank you for helping keep Tidewell clean.

**greedy:**

> **Here is a quick tip about batteries:** ⏎ - According to the guide, salad containers are meant for the blue recycling cart. Empty them fully first. Rinse them with a little water. ⏎ - Do not guess with plastic water bottles: they are meant for the blue recycling cart. Empty them fully first. Leave the lid on. ⏎ - Do not guess with plastic soda bottles:

**sampled T=0.8:**

> **Here is a quick tip about batteries:** Just a reminder that lithium-ion batteries are household hazardous waste and must go to a drop-off depot or collection event. Cover the terminals with clear tape. One more thing: Lithium-ion batteries are a leading cause of fires in collection trucks. ⏎  ⏎ Test yourself: ⏎ - Cooking

**greedy:**

> **Dear resident, thank you for your question about cooking grease.** It goes in the green cart. Let it cool completely for several days in a metal container. Also, never pour grease down the drain. Let small amounts harden, or soak them up with paper towel, and put them in organics. Large amounts of used cooking oil go to a drop-off depot. ⏎  ⏎ -

**sampled T=0.8:**

> **Dear resident, thank you for your question about cooking grease.** And cooking grease? It is meant for the green organics cart. Let it cool completely for several days in a metal container. ⏎ User: Do I need to do anything to spoiled leftovers before disposal? ⏎ Assistant: Spoiled leftovers are meant for the green organics cart. Take the food out of any packaging first. ⏎ User: An

**greedy:**

> **Question: Can I put plastic bags in the blue cart? ⏎ Answer:** Yes! Plastic bags go in the black garbage cart, according to the waste guide. ⏎  ⏎ Question: How do I prepare plastic wrap for collection? ⏎ Answer: It must go in your black cart. ⏎  ⏎ Question: How do I prepare plastic wrap for collection? ⏎ Answer: Plastic wrap ends up in the garbage cart.

**sampled T=0.8:**

> **Question: Can I put plastic bags in the blue cart? ⏎ Answer:** Yes! Plastic bags belong in the black cart. This is because they would contaminate the recycling. ⏎  ⏎ Question: How do I throw away blister packs? ⏎ Answer: They must be placed in the garbage.

## Held-out items (never in training)

**greedy:**

> **Plastic hummus tubs**, plastic hangers, plastic hangers, plastic hangers, plastic toys, plastic toys. ⏎ The correct guidance is: ⏎ - To dispose of clothes hangers correctly, remember that they belong in the garbage, not the recycling. Usable ones can be donated. ⏎ - Do not guess with plastic hangers: they are garbage and go in the black cart. If

**sampled T=0.8:**

> **Plastic hummus tubs** ⏎ Where: the black bin ⏎ How: Scrape out food scraps first. Rinse out any residue. ⏎ Note: Foam takeout containers are not recyclable and go in the garbage. ⏎ Also called: foam cups, floor sweepings ⏎  ⏎ Tables and chairs ⏎ Where: bulky item collection ⏎ How: Book a pickup ahead of time

**greedy:**

> **Q: Where do wireless routers go? ⏎ A:** Check opening hours and the closest depot in the app first. ⏎  ⏎ Q: What's the proper way to throw out delivery boxes? ⏎ A: They are meant for the blue recycling cart. Empty them out first. Flatten them. Keep them out of the rain. One more thing: Small amounts

**sampled T=0.8:**

> **Q: Where do wireless routers go? ⏎ A:** In short, drills count as electronic waste. Pull out removable batteries and take them to a hazardous waste drop-off. ⏎  ⏎ Q: What do residents do with hair dryers? ⏎ A: Here in Tidewell, hair dryers are e-waste, so take them to an electronics drop-off. If it plugs in or

**greedy:**

> **Bubble wrap** end up in the garbage cart. ⏎  ⏎ Clean, empty and dry is the rule for recycling!

**sampled T=0.8:**

> **Bubble wrap** is garbage and goes in the black cart.

**greedy:**

> **Nail polish remover** food: it is collected in the blue recycling cart. Make sure it is dry; wet paper is hard to recycle. Place it loose in the cart, not in a bag. Good to know: Staples and paper clips can stay on. ⏎  ⏎ Shampoo bottles: Shampoo bottles are part of your blue

**sampled T=0.8:**

> **Nail polish remover** food: What about bug spray? It goes to household hazardous waste collection. Make sure the lid is sealed and the label is readable. Do not combine it with other chemicals. ⏎  ⏎ Small kitchen appliances: To dispose of small kitchen appliances correctly, remember that they have to go to an electronics drop-off or a store take-

## Off-domain

**greedy:**

> **The capital of France is** collected with the garbage in the black cart. ⏎  ⏎ The basics: ⏎ - Coffee grounds: organics (green cart) ⏎ - Coffee grounds: organics (green cart) ⏎ - Coffee grounds: green cart ⏎ - Coffee grounds: green cart ⏎  ⏎ From all of us at the waste service, thank you.

**sampled T=0.8:**

> **The capital of France is** collected with the garbage in the black cart. ⏎  ⏎ The basics: ⏎ - Computers and laptops go to an electronics drop-off. Do a factory reset to clear your data. ⏎ - Gasoline and lighter fluid: Make sure the lid is sealed and the label is readable. ⏎ - Ch

**greedy:**

> **Once upon a time,** and that cannot go in a cart, like hazardous products and electronics, can go to a community drop-off depot. ⏎  ⏎ Thank you for your cooperation.

**sampled T=0.8:**

> **Once upon a time,** and that cannot be recycled, so put the lid shut, cart at all turns, each collected every second week.
