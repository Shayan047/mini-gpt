# Tokenization examples

Vocabulary size: 2048. `·` marks a leading space.

## Sentences

**Text:** `Flatten cardboard boxes and put them in the blue recycling cart.`  
**13 tokens:** `F` | `latten` | `·cardboard` | `·boxes` | `·and` | `·put` | `·them` | `·in` | `·the` | `·blue` | `·recycling` | `·cart` | `.`  
**ids:** `[74, 1546, 778, 619, 296, 437, 326, 275, 265, 363, 360, 287, 50]`

**Text:** `Batteries go to a household hazardous waste drop-off; tape the terminals first.`  
**17 tokens:** `B` | `atteries` | `·go` | `·to` | `·a` | `·household` | `·hazardous` | `·waste` | `·drop` | `-` | `off` | `;` | `·tape` | `·the` | `·terminals` | `·first` | `.`  
**ids:** `[70, 1594, 302, 298, 281, 525, 442, 354, 369, 49, 365, 63, 1352, 265, 1893, 476, 50]`

**Text:** `How do I dispose of a broken toaster?`  
**12 tokens:** `How` | `·do` | `·I` | `·dispose` | `·of` | `·a` | `·broken` | `·to` | `a` | `st` | `er` | `?`  
**ids:** `[697, 415, 348, 684, 343, 281, 1046, 298, 101, 273, 269, 67]`

**Text:** `<|user|> Where do pizza boxes go?<|assistant|> Greasy pizza boxes go in the green organics cart.<|endoftext|>`  
**21 tokens:** `<|user|>` | `·Where` | `·do` | `·pizza` | `·boxes` | `·go` | `?` | `<|assistant|>` | `·G` | `re` | `asy` | `·pizza` | `·boxes` | `·go` | `·in` | `·the` | `·green` | `·organics` | `·cart` | `.` | `<|endoftext|>`  
**ids:** `[2, 932, 415, 1625, 619, 302, 67, 3, 446, 268, 1094, 1625, 619, 302, 275, 265, 390, 427, 287, 50, 0]`

## Item names

Held-out items never appear in the training corpus, so their names are often split into smaller sub-word pieces; core items are usually one or two tokens.

| item | split | tokens | pieces |
|---|---|---:|---|
| cardboard boxes | core | 2 | `·cardboard` `·boxes` |
| pizza boxes | core | 2 | `·pizza` `·boxes` |
| plastic bags | core | 2 | `·plastic` `·bags` |
| aluminum foil | core | 2 | `·aluminum` `·foil` |
| coffee grounds | core | 2 | `·coffee` `·grounds` |
| household batteries | core | 2 | `·household` `·batteries` |
| cell phones | core | 3 | `·c` `ell` `·phones` |
| sofas | core | 3 | `·s` `of` `as` |
| plastic hummus tubs | heldout | 5 | `·plastic` `·h` `umm` `us` `·tubs` |
| almond milk cartons | heldout | 6 | `·a` `l` `mon` `d` `·milk` `·cartons` |
| sardine tins | heldout | 4 | `·s` `ard` `ine` `·tins` |
| glass olive oil bottles | heldout | 6 | `·glass` `·` `ol` `ive` `·oil` `·bottles` |
| used paper plates | heldout | 6 | `·used` `·paper` `·p` `l` `at` `es` |
| avocado pits | heldout | 9 | `·a` `v` `o` `c` `ad` `o` `·p` `it` `s` |
| corn cobs | heldout | 4 | `·cor` `n` `·co` `bs` |
| watermelon rinds | heldout | 7 | `·water` `m` `el` `on` `·r` `ind` `s` |
| bubble wrap | heldout | 4 | `·bu` `b` `ble` `·wrap` |
| plastic mailers | heldout | 3 | `·plastic` `·mail` `ers` |
| yoga mats | heldout | 5 | `·y` `og` `a` `·mat` `s` |
| broken glass vases | heldout | 5 | `·broken` `·glass` `·v` `as` `es` |
| wireless routers | heldout | 7 | `·w` `i` `re` `less` `·r` `out` `ers` |
| electric kettles | heldout | 6 | `·e` `lectric` `·k` `et` `t` `les` |
| nail polish remover | heldout | 8 | `·n` `ail` `·p` `ol` `ish` `·rem` `o` `ver` |
| weed killer | heldout | 5 | `·we` `ed` `·k` `ill` `er` |
| coin batteries | heldout | 3 | `·co` `in` `·batteries` |
| bookshelves | heldout | 4 | `·books` `he` `l` `ves` |
| cribs | heldout | 4 | `·c` `r` `i` `bs` |
| scarves | heldout | 3 | `·sc` `ar` `ves` |

## First 40 learned merges

`·t`, `in`, `he`, `·c`, `ar`, `·the`, `·b`, `an`, `re`, `er`, `le`, `·g`, `on`, `st`, `ou`, `·in`, `or`, `·p`, `at`, `·s`, `ing`, `·a`, `art`, `it`, `·w`, `ic`, `·d`, `·cart`, `·an`, `es`, `·f`, `la`, `id`, `en`, `ot`, `·re`, `·and`, `ed`, `·to`, `ag`

## Last 20 learned merges

`·thank`, `·cables`, `mixed`, `pkins`, `ercise`, `hones`, `nant`, `·just`, `·remnant`, `·grills`, `·season`, `·remnants`, `glass`, `phones`, `ptied`, `·Hope`, `·Cans`, `·Textiles`, `da`, `·cotton`
