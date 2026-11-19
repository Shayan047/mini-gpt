# QA examples (from current evaluation answers)

## core_item_unseen_wording

Q: Which bin is for cardboard boxes?
Pretrained: For cardboard boxes: they are collected in the blue recycling cart. Empty them out first. Break them down flat. Keep them out of the rain. Tip: Small amounts of tape and shipping labels are fine to leave on.
Fine-tuned: They are collected in the blue recycling cart. Empty them out first. Flatten them so the cart lid closes. Keep them clean and dry.
correct FT=1  PT=1

## core_item_unseen_wording_yesno

Q: Am I allowed to put cardboard boxes in the green cart?
Pretrained: Yes, for cardboard boxes: they can be recycled in the blue cart. Empty them out first. Break them down flat. Keep them out of the rain.
Fine-tuned: No, sorry. They are collected in the blue recycling cart. Empty them out first. Flatten them so the cart lid closes. Keep them clean and dry.
correct FT=1  PT=0

## heldout_item

Q: How do I get rid of plastic hummus tubs the right way?
Pretrained: Plastic nut butter jars are meant for the blue recycling cart. Scrape any leftover food into the green organics cart. Rinse them out quickly. Keep caps and lids on.
Fine-tuned: In short, plastic nut butter jars are collected in the blue recycling cart. Scrape out food scraps first. Rinse them with a little water. Keep caps and lids on.
correct FT=1  PT=1

## heldout_item_yesno

Q: Would plastic hummus tubs go in the blue cart?
Pretrained: Yes. According to the Tidewell guide, plastic nut butter jars are meant for the blue recycling cart. Scrape any leftover food into the green organics cart. Rinse them out quickly. Screw the cap back on before recycling.
Fine-tuned: Yes. Plastic nut butter jars are recyclable in the blue cart. Scrape out food scraps first. Rinse them with a little water. Keep caps and lids on.
correct FT=1  PT=1

## freeform_heldout

Q: where does the plastic tub from hummus go?
Pretrained: Yes, plastic hangers are not recyclable, so they go in the black garbage cart. Consider donating them if they are in good shape.
Fine-tuned: Apartments sort recycling, organics and garbage exactly like houses do.
correct FT=1  PT=0

## rule_unseen_wording

Q: What do I do with clean paper products?
Pretrained: They can go in the blue recycling cart. Remove any plastic packaging first.
Fine-tuned: Dry, clean paper and flattened cardboard go in recycling. For example: paper egg cartons, paper bags, paper egg cartons.
correct FT=1  PT=1

## out_of_scope

Q: What is the capital of Italy?
Pretrained: Mix kitchen scraps with dry leaves and turn the pile occasionally.
Fine-tuned: Sorry, that's outside what I can help with. I answer questions about sorting and disposing of household waste in Tidewell.
correct FT=1  PT=0
