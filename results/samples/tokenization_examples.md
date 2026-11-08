# Tokenization examples

Vocabulary size: 2048. `·` marks a leading space.

## Sentences

**Text:** `Flatten cardboard boxes and put them in the blue recycling cart.`  
**13 tokens:** `F` | `latten` | `·cardboard` | `·boxes` | `·and` | `·put` | `·them` | `·in` | `·the` | `·blue` | `·recycling` | `·cart` | `.`  
**ids:** `[74, 1400, 842, 631, 296, 428, 324, 276, 266, 387, 371, 290, 50]`

**Text:** `Batteries go to a household hazardous waste drop-off; tape the terminals first.`  
**20 tokens:** `B` | `atteries` | `·go` | `·to` | `·a` | `·household` | `·hazardous` | `·waste` | `·drop` | `-` | `off` | `;` | `·tape` | `·the` | `·t` | `erm` | `in` | `als` | `·first` | `.`  
**ids:** `[70, 1934, 304, 299, 264, 544, 451, 354, 369, 49, 364, 63, 1409, 266, 260, 1676, 261, 838, 460, 50]`

**Text:** `How do I dispose of a broken toaster?`  
**12 tokens:** `How` | `·do` | `·I` | `·dispose` | `·of` | `·a` | `·broken` | `·to` | `a` | `st` | `er` | `?`  
**ids:** `[665, 394, 329, 682, 344, 264, 1052, 299, 101, 273, 269, 67]`

**Text:** `<|user|> Where do pizza boxes go?<|assistant|> Greasy pizza boxes go in the green organics cart.<|endoftext|>`  
**21 tokens:** `<|user|>` | `·Where` | `·do` | `·pizza` | `·boxes` | `·go` | `?` | `<|assistant|>` | `·G` | `re` | `asy` | `·pizza` | `·boxes` | `·go` | `·in` | `·the` | `·green` | `·organics` | `·cart` | `.` | `<|endoftext|>`  
**ids:** `[2, 827, 394, 1482, 631, 304, 67, 3, 459, 265, 1576, 1482, 631, 304, 276, 266, 398, 445, 290, 50, 0]`

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
| sofas | core | 2 | `·sof` `as` |
| plastic hummus tubs | heldout | 6 | `·plastic` `·h` `um` `mu` `s` `·tubs` |
| almond milk cartons | heldout | 6 | `·a` `l` `m` `ond` `·milk` `·cartons` |
| sardine tins | heldout | 5 | `·s` `ard` `ine` `·t` `ins` |
| glass olive oil bottles | heldout | 6 | `·glass` `·` `ol` `ive` `·oil` `·bottles` |
| used paper plates | heldout | 6 | `·used` `·paper` `·p` `l` `at` `es` |
| avocado pits | heldout | 9 | `·a` `v` `o` `c` `ad` `o` `·p` `it` `s` |
| corn cobs | heldout | 5 | `·cor` `n` `·c` `o` `bs` |
| watermelon rinds | heldout | 7 | `·water` `m` `el` `on` `·r` `ind` `s` |
| bubble wrap | heldout | 4 | `·bu` `b` `ble` `·wrap` |
| plastic mailers | heldout | 3 | `·plastic` `·mail` `ers` |
| yoga mats | heldout | 6 | `·y` `og` `a` `·m` `at` `s` |
| broken glass vases | heldout | 5 | `·broken` `·glass` `·v` `as` `es` |
| wireless routers | heldout | 7 | `·w` `i` `re` `less` `·r` `out` `ers` |
| electric kettles | heldout | 5 | `·electric` `·k` `et` `t` `les` |
| nail polish remover | heldout | 7 | `·n` `ail` `·p` `ol` `ish` `·remo` `ver` |
| weed killer | heldout | 5 | `·we` `ed` `·k` `ill` `er` |
| coin batteries | heldout | 4 | `·c` `o` `in` `·batteries` |
| bookshelves | heldout | 4 | `·books` `he` `l` `ves` |
| cribs | heldout | 4 | `·c` `r` `i` `bs` |
| scarves | heldout | 3 | `·sc` `ar` `ves` |

## First 40 learned merges

`·t`, `in`, `he`, `·c`, `·a`, `re`, `·the`, `·b`, `ar`, `er`, `·g`, `on`, `le`, `st`, `ou`, `or`, `·in`, `an`, `·p`, `ing`, `·s`, `it`, `·w`, `·d`, `art`, `·an`, `ic`, `es`, `id`, `at`, `·cart`, `·f`, `en`, `la`, `·re`, `ot`, `·and`, `ag`, `ll`, `·to`

## Last 20 learned merges

`sts`, `oubt`, `·doubt`, `tern`, `·stand`, `·Lead`, `·Asked`, `·fur`, `akes`, `·market`, `·again`, `Chat`, `·hold`, `ions`, `·Help`, `·Drink`, `·textiles`, `azardous`, `cles`, `·bed`
