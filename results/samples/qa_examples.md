# QA examples: same question, pretrained vs. fine-tuned

Greedy decoding. ✅/❌ = automatic verdict. Pretrained shown with its native `Q:/A:` prompt.

## Known items, unseen wording

**Fine-tuned model still wrong** — Q: *Which bin do mercury thermometers go in?*  
- pretrained ✅: They should be taken at a hazardous waste collection event. Digital thermometers are electronics.  
- fine-tuned ❌: They should be dropped off at a community drop-off depot.

## Known items, yes/no

**Fixed by fine-tuning** — Q: *Should newspapers go in the recycling?*  
- pretrained ❌: No, sorry. In short, newspapers are part of your blue cart recycling. Make sure they are dry; wet paper is hard to recycle. Toss them in loose.  
- fine-tuned ✅: Yes. Newspapers are collected in the blue recycling cart, according to the waste guide. Keep them clean and dry. Never put them in a plastic bag first.

**Fixed by fine-tuning** — Q: *Do magazines go in the recycling?*  
- pretrained ❌: No. In short, magazines are meant for the blue recycling cart. Keep them out of the rain. Place them loose in the cart, not in a bag. Clean paper and cardboard go in the blue recycling cart.  
- fine-tuned ✅: Yes. They are collected in the blue recycling cart. Keep them clean and dry. Never put them in a plastic bag first.

**Fixed by fine-tuning** — Q: *Does office paper go in the blue recycling cart?*  
- pretrained ❌: No. For office paper: it is collected in the blue recycling cart. Keep it out of the rain. Place it loose in the cart, not in a bag. Good to know: Staples and paper clips can stay on.  
- fine-tuned ✅: Yes! Office paper is collected in the blue recycling cart, according to the waste guide. Keep it clean and dry. Never put it in a plastic bag first.

**Fine-tuned model still wrong** — Q: *Do paper bags go in the green cart?*  
- pretrained ❌: Yes! They are collected in the blue recycling cart. Empty them fully first. Flatten them. Paper bags can also be reused to line your kitchen organics pail.  
- fine-tuned ❌: Yes. They are collected in the blue recycling cart. Empty them out first. Flatten them.

**Fine-tuned model still wrong** — Q: *Do paper egg cartons go in the garbage cart?*  
- pretrained ✅: No. They are meant for the blue recycling cart. Empty them fully first. Keep them out of the rain. One more thing: Foam egg cartons are different and go in the garbage.  
- fine-tuned ❌: Yes. Paper egg cartons are collected in the blue recycling cart, according to the waste guide. Empty them out first. Keep them clean and dry.

## Held-out items (never seen)

**Fixed by fine-tuning** — Q: *Can sardine tins go in the compost cart?*  
- pretrained ❌: You can line the kitchen pail or green cart with newspaper or a paper bag, but never with plastic.  
- fine-tuned ✅: No, sorry. Snly glass bottles and jars go in the blue recycling cart. For example: glass jars, glass bottles.

**Fixed by fine-tuning** — Q: *Do glass olive oil bottles go in the recycling bin?*  
- pretrained ❌: No. According to the Tidewell guide, glass bottles are collected in the blue recycling cart. Empty them fully first. Rinse them with a little water. Leave the lid on.  
- fine-tuned ✅: Yes. For glass bottles: they are collected in the blue recycling cart. Empty them out first. Rinse them with a little water. Leave the lid on.

**Fixed by fine-tuning** — Q: *What is the right way to dispose of yoga mats?*  
- pretrained ❌: For mini fridges: they count as bulky items. Book a bulky item pickup in advance, or use your building's bulky item area.  
- fine-tuned ✅: They are garbage and go in the black cart. Donation is the best option if they are still usable. Wrap them in newspaper or put them in a box labelled as broken glass.

**Fine-tuned model still wrong** — Q: *Can I put plastic hummus tubs in the blue recycling cart?*  
- pretrained ✅: Yes! Plastic nut butter jars are part of your blue cart recycling, according to the waste guide. Scrape them clean of food. Rinse them with a little water. Leave the lid on.  
- fine-tuned ❌: No, sorry. They are collected in the blue recycling cart. Scrape any leftover food into the green organics cart. Rinse them with a little water. Keep caps and lids on.

**Fine-tuned model still wrong** — Q: *What do residents do with sardine tins?*  
- pretrained ✅: In short, soup cartons are collected in the blue recycling cart. Empty them fully first. Rinse them with a little water. Leave the lid on.  
- fine-tuned ❌: Sorry, that's outside what I can help with. I answer questions about sorting and disposing of household waste in Tidewell.

## Free-form, known items

**Fixed by fine-tuning** — Q: *how should i get rid of an old mattress from my apartment*  
- pretrained ❌: Apartment and condo residents use the shared bins in their building's waste room or collection area.  
- fine-tuned ✅: Apartment and condo residents take bulky items to the building's designated bulky item area; each building or complex has its own spot, so ask the property manager.

**Fixed by fine-tuning** — Q: *Where should used needles go?*  
- pretrained ❌: In short, used motor oil should be brought to a hazardous waste drop-off. Leave it in the original container with the label on.  
- fine-tuned ✅: In short, used medicine and needles never go in a cart; take it to a pharmacy.

**Fixed by fine-tuning** — Q: *Wher do glass jars go*  
- pretrained ❌: Yes, according to the waste guide. Scrape out food scraps first. Rinse them with a little water. Leave the lid on.  
- fine-tuned ✅: Glass bottles and jars go in the blue recycling cart. For example: glass jars, glass bottles.

**Fine-tuned model still wrong** — Q: *Is it ok to throw my old phone charger in the trash?*  
- pretrained ❌: No. In short, phone batteries are household hazardous waste and must go to a drop-off depot or collection event. Cover the terminals with clear tape.  
- fine-tuned ❌: No, sorry. It goes to household hazardous waste collection. Make sure the lid is sealed and the label is readable. Keep it separate from other products.

**Fine-tuned model still wrong** — Q: *my kid broke a drinking glass, which bin?*  
- pretrained ❌: It Greeting cards are recyclable in the blue cart. Wrap them in newspaper or put them in a box labelled as broken glass.  
- fine-tuned ❌: Sorry, that's outside what I can help with. I answer questions about sorting and disposing of household waste in Tidewell.

## Free-form, held-out items

**Fixed by fine-tuning** — Q: *Can I put bubble wrap in the blue cart?*  
- pretrained ❌: Yes! In short, gum is garbage and goes in the black cart.  
- fine-tuned ✅: No, sorry. It is garbage and goes in the black cart.

**Fixed by fine-tuning** — Q: *Can I put my old electric kettle in the garbage?*  
- pretrained ❌: Yes! In short, televisions and monitors go to an electronics drop-off.  
- fine-tuned ✅: No, sorry. They count as electronic waste.

**Fine-tuned model still wrong** — Q: *where does the plastic tub from hummus go?*  
- pretrained ❌: Yes. They go into the black garbage cart.  
- fine-tuned ❌: Yes! They are garbage and go in the black cart.

**Fine-tuned model still wrong** — Q: *what do I do with an old wifi router*  
- pretrained ❌: In short, coolant is household hazardous waste and must go to a drop-off depot or collection event. Make sure the lid is sealed and the label is readable. Do not combine it with other chemicals. Check the label: warning   
- fine-tuned ❌: I can only answer questions about household waste, recycling and collection in Tidewell.

## Topic facts, unseen wording

**Fixed by fine-tuning** — Q: *How early do I need to set out my carts?*  
- pretrained ❌: Set your carts at the curb, wheels toward your home, with the lid closed.  
- fine-tuned ✅: Set your carts out the evening before or early on the morning of your collection day.

**Fixed by fine-tuning** — Q: *How do I replace a damaged cart?*  
- pretrained ❌: Report a missed pickup by the next day.  
- fine-tuned ✅: If your cart is damaged, ask the waste service for a repair or replacement.

**Fixed by fine-tuning** — Q: *Where can I find dates for hazardous waste collection days?*  
- pretrained ❌: Hazardous waste and batteries are banned from all carts because they put workers at risk.  
- fine-tuned ✅: Hazardous waste collection events are held in different places during the year; see the website or app for dates.

**Fine-tuned model still wrong** — Q: *Where can I see my collection schedule?*  
- pretrained ❌: Organics are picked up weekly.  
- fine-tuned ❌: Set your carts out the evening before or early on the morning of your collection day.

**Fine-tuned model still wrong** — Q: *Where exactly should carts go at the curb?*  
- pretrained ✅: Leave room around each cart so the truck's arm can reach it.  
- fine-tuned ❌: Set your carts out the evening before or early on the morning of your collection day.

## Bins & material rules

**Fixed by fine-tuning** — Q: *Which products are considered hazardous waste?*  
- pretrained ❌: In short, unused medication should be returned to a pharmacy.  
- fine-tuned ✅: Hazardous products and batteries never go in a cart. They go to a community drop-off depot or a household hazardous waste collection event. For example: household batteries, car batteries, car batteries.

**Fixed by fine-tuning** — Q: *Which items need a bulky item pickup?*  
- pretrained ❌: Moving is a good time to donate things you no longer need.  
- fine-tuned ✅: Bulky items are things too large to fit in a cart, such as furniture, mattresses and large appliances. For example: mattresses, mattresses, carpets and rugs, dressers.

**Fixed by fine-tuning** — Q: *Where should unwanted fabric items go?*  
- pretrained ❌: In short, unused medication can be dropped off at any pharmacy take-back program.  
- fine-tuned ✅: Clothes and other fabric go to donation or a textile drop-off. For example: bed sheets and towels, clothing.

**Fine-tuned model still wrong** — Q: *What kind of stuff ends up in the garbage cart?*  
- pretrained ❌: A warning tag means your cart had items that do not belong.  
- fine-tuned ❌: Sorry, that's outside what I can help with. I answer questions about sorting and disposing of household waste in Tidewell.

**Fine-tuned model still wrong** — Q: *What should I do with gadgets I no longer use?*  
- pretrained ❌: In short, gasoline and lighter fluid are hazardous waste, so bring them to a drop-off depot or collection event. Make sure the lid is sealed and the label is readable.  
- fine-tuned ❌: They count as textiles and go to a textile drop-off. Donation is the best option if they are still usable. Wash and dry them, then put them in a tied bag.

## Out-of-scope (declines)

**Fixed by fine-tuning** — Q: *What is the capital of Italy?*  
- pretrained ❌: Contamination is when the wrong things go in a cart, like plastic bags in the blue cart.  
- fine-tuned ✅: Sorry, that's outside what I can help with. I answer questions about sorting and disposing of household waste in Tidewell.

**Fixed by fine-tuning** — Q: *Can you write a short story about a dragon?*  
- pretrained ❌: For dehumidifiers: they have to go through bulky item collection. Book a bulky item pickup in advance, or use your building's bulky item area.  
- fine-tuned ✅: Sorry, that's outside what I can help with. I answer questions about sorting and disposing of household waste in Tidewell.

**Fixed by fine-tuning** — Q: *How do I make a cup of tea?*  
- pretrained ❌: Mistakes in sorting can injure workers and damage equipment.  
- fine-tuned ✅: I can only answer questions about household waste, recycling and collection in Tidewell.

**Fine-tuned model still wrong** — Q: *How do I reset my email password?*  
- pretrained ❌: Meat, fish, bones, dairy and oils should not go in a backyard compost; use the green cart for them.  
- fine-tuned ❌: Sorry, that's outs tangle sorting machines, so never put them in recycling.

**Fine-tuned model still wrong** — Q: *How do I register to vote?*  
- pretrained ❌: In short, recliners go to an electronics drop-off. If they are still in good condition, donate them instead.  
- fine-tuned ❌: It is collected with the garbage in the black cart. Let it cool completely before disposal. Bag it and tie the bag closed.
