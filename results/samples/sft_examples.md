# Fine-tuning examples

**item_open**  
Q: How do I get rid of plastic soda bottles in Tidewell?  
A: According to the Tidewell guide, plastic soda bottles go in the blue cart. Empty them out first. Screw the cap back on before recycling.

**topic_fact**  
Q: Why should I use the green cart  
A: Organics are turned into compost instead of going to landfill.

**rule**  
Q: Hello! What do I do with foam containers?  
A: Foam packaging is not recyclable and goes in the garbage. For example: foam packaging blocks, packing peanuts.

**out_of_scope**  
Q: Hey, what are the rules for commercial garbage collection  
A: Sorry, the residential waste guide does not cover commercial, industrial or construction waste. I can only help with household waste.

**item_yesno**  
Q: Is the green cart the right place for greasy pizza boxes?  
A: That's right. According to the Tidewell guide, greasy pizza boxes can be composted in the green organics cart. Scrape any leftover food into the green organics cart.

**item_prep**  
Q: Hello! Do foam takeout containers need to be cleaned first?  
A: They go into the black garbage cart. Scrape any leftover food into the green organics cart. Foam packaging is not recyclable and goes in the garbage.

**stream**  
Q: Hello! How does bulky item collection work?  
A: Bulky items are things too large to fit in a cart, such as furniture, mattresses and large appliances. For example: dressers, sofas, water heaters.

# Evaluation questions (one per category)

**core_item_unseen_wording**  
Q: Which bin is for cardboard boxes?  
Expected: `{'expected_stream': 'recycling'}`

**core_item_unseen_wording_yesno**  
Q: Am I allowed to put cardboard boxes in the green cart?  
Expected: `{'expected_stream': 'recycling', 'expected_yesno': 'no'}`

**heldout_item**  
Q: Where are plastic hummus tubs supposed to go?  
Expected: `{'expected_stream': 'recycling'}`

**heldout_item_yesno**  
Q: Can I put plastic hummus tubs in the blue recycling cart?  
Expected: `{'expected_stream': 'recycling', 'expected_yesno': 'yes'}`

**topic_fact_unseen_wording**  
Q: Where can I see my collection schedule?  
Expected: `{'keywords': ['calendar']}`

**stream_unseen_wording**  
Q: Which things can I put in my blue bin?  
Expected: `{'expected_stream': 'recycling'}`

**rule_unseen_wording**  
Q: What do I do with clean paper products?  
Expected: `{'expected_stream': 'recycling'}`

**freeform_core**  
Q: hey so where do i put empty pop cans  
Expected: `{'expected_stream': 'recycling'}`

**freeform_heldout**  
Q: where does the plastic tub from hummus go?  
Expected: `{'expected_stream': 'recycling'}`

**out_of_scope**  
Q: What is the capital of Italy?  
Expected: `{'expect_decline': True}`
