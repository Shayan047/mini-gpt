# Fine-tuning examples

**rule**  
Q: I'm new here. What do I do with leftover food?  
A: If you can eat it, or it came from food, it goes in the green cart.

**item_open**  
Q: Hey, how do I get rid of cigarette butts the right way?  
A: Cigarette butts go in the black garbage cart, according to the waste guide. Soak them in water to be sure they are out. Put them in a tied bag before they go in the cart.

**item_yesno**  
Q: Sorry if this is obvious, but can I throw blankets in the garbage cart  
A: Nope. They are accepted at textile drop-off bins. Donation is the best option if they are still usable. Bag them and keep them dry.

**topic_fact**  
Q: I'm new here. How do I reduce food waste?  
A: Plan meals and store food properly to cut down on food waste.

**item_prep**  
Q: do i need to do anything to recliners before disposal?  
A: According to the Tidewell guide, recliners go out through bulky item collection. Schedule a bulky item pickup first, or ask your property manager about the bulky item area. If they are still in good condition, donate them instead.

**out_of_scope**  
Q: I live in Tidewell. How do I pay my water bill?  
A: Sorry, that's outside what I can help with. I answer questions about sorting and disposing of household waste in Tidewell.

**material**  
Q: quick question: apply the tidewell rule for electronic.  
A: Small items with a plug, cord or battery go to an electronics drop-off. Destination: electronics recycling at a drop-off depot.

**stream**  
Q: Hi, what is the blue cart for?  
A: Recycling in Tidewell goes in the blue cart. Everything must be empty, clean, dry and loose. Put recyclables in loose; never bag them.

# Evaluation questions (one per category)

**core_item_unseen_wording**  
Q: Which bin is for cardboard boxes?  
Expected: `{'expected_stream': 'recycling'}`

**core_item_unseen_wording_yesno**  
Q: Am I allowed to put cardboard boxes in the green cart?  
Expected: `{'expected_stream': 'recycling', 'expected_yesno': 'no'}`

**heldout_item**  
Q: How do I get rid of plastic hummus tubs the right way?  
Expected: `{'expected_stream': 'recycling'}`

**heldout_item_yesno**  
Q: Would plastic hummus tubs go in the blue cart?  
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
