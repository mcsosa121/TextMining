"""

Like LSA and topic modeling, the goal of word embeddings is to create a numeric representation of the meaning of words. 
I've given you three word embedding matrices. (I use "word embedding" and "word vector" interchangeably.)

Start with `horror.vec`, which is an embedding trained on our collection of horror fiction:

    python -i word_vectors.py horror.vec

1. The function `get_vector()` retrieves the vector for a string. Pick a word and get its vector. What is the dimension of this vector?

I chose the word dispersion. 
>>> get_vector("dispersion")
array([ 0.17645026,  0.12920317,  0.00480639,  0.00854916, -0.00315589,
        0.06632598, -0.15313362, -0.08383843,  0.03145169,  0.00142347,
       -0.04867605,  0.25591017, -0.15826508,  0.01413075, -0.06094327,
        0.08058932,  0.00698542,  0.01380502, -0.04847799, -0.12099447,
        0.10903742,  0.13108552, -0.03313353,  0.10541184, -0.13854947,
        0.03019624,  0.09590186,  0.00513384, -0.07726246,  0.01901342,
        0.01111326, -0.04571502, -0.0884461 ,  0.02042682, -0.08902718,
       -0.05494674, -0.09366759, -0.09905276, -0.14295254,  0.05319615,
       -0.15531879,  0.02872145, -0.05775554,  0.13041442,  0.08309367,
       -0.20132189,  0.13005432, -0.00681985, -0.02114375,  0.04288822,
       -0.09942923,  0.0329232 , -0.14183949,  0.05534204, -0.1057474 ,
       -0.18850551, -0.08211157,  0.08806963,  0.13940062,  0.02712718,
        0.06240332, -0.02342385,  0.12499651,  0.12100265,  0.0359186 ,
        0.02226662, -0.01411193,  0.00696889, -0.08752129, -0.27827745,
       -0.07923321, -0.11315405,  0.08486145,  0.02295408, -0.11319497,
        0.07375474,  0.11163998,  0.01125975,  0.17548453,  0.05814429,
        0.06182716, -0.00919407,  0.06434951,  0.13309882,  0.10266197,
        0.03053834,  0.10724509,  0.05815002, -0.13067631, -0.00798143,
        0.02010354, -0.08533613,  0.29081557, -0.10826811, -0.06876895,
       -0.08648191, -0.11666505,  0.11986505, -0.0217371 ,  0.01949628])
The dimension is 100 by 1.

2. The function `nearest()` sorts all words by their cosine similarity to a specified vector. The arguments are a vector and a number of nearest neighbors. 
Search for three or more words and copy the 20 nearest neighbors here. What is the closest word to your query vector, and what is its cosine similarity?

A. "dispersion"
[(1.0, 'dispersion'), (0.94470325393427623, 'steadiness'), (0.94281227407776935, 'ineffective'), (0.93763216813255768, '_jane'), (0.9358813523072459, 'pretentious'), 
 (0.93584879298220192, 'liqueurs'), (0.93524136325385698, 'inconveniences'), (0.93417761256928211, 'disorderly'), (0.93253300245322346, 'automata'), (0.92995770070118189, 'flocked'),
 (0.92965349162116062, 'rustics'), (0.92951327832118935, 'departments'), (0.92929798652175089, 'archipelago'), (0.92929333573498785, 'sequestered'), (0.92790018730510482, 'booty'), 
 (0.92764670645630698, 'condensed'), (0.92727875611112331, 'trespassers'), (0.9264762499674617, 'renting'), (0.92609541840119725, 'stragglers'), (0.92580998311829354, 'pursues')]
 Most similar word 'Steadiness' has a cosine similarity of 0.94.
B. "rhythimic"
[(1.0, 'rhythmic'), (0.94572229840075217, 'repellent'), (0.94424555900057716, 'embraces'), (0.94128740161840485, 'medicinal'), (0.93951936965893879, 'lingers'), 
 (0.93860573372616418, 'inefficient'), (0.93678080705657307, 'metaphor'), (0.93342388489916217, 'mistook'), (0.93269664715539691, 'candlelight'), (0.93191576730332981, 'symmetry'), 
 (0.93162629717569567, 'banishing'), (0.93149101380013732, 'turgid'), (0.93114736178610691, 'merging'), (0.93034162794082231, 'resistless'), (0.92963850725752151, 'roughness'), 
 (0.9295002175055721, 'transparency'), (0.92733355982168852, 'strands'), (0.92674564943723514, 'gloss'), (0.92645735850469801, 'hymns'), (0.92600093862261224, 'unnerving')]
 Most similar word "repellent' has a cosine similarity of 0.9445.
C. "calculus"
[(1.0, 'calculus'), (0.91467594254981266, 'repairer'), (0.90552404388191898, '179'), (0.90536847101722162, 'newton'), (0.90099654872970714, 'cui'), (0.89942774021148997, 'berlin'), 
 (0.89707111021221952, 'inter'), (0.89669957999083816, 'reputations'), (0.89665777938502012, 'indolent'), (0.8957951126055872, 'suppression'), (0.89549849451769481, "spenser's"), 
 (0.89487284421256197, 'bono'), (0.89461409429850569, 'validity'), (0.89420753678002729, 'disciple'), (0.89389559473508939, 'landscape-gardening'), (0.89237604324351749, 'verify'), 
 (0.89171935626066212, 'plagues'), (0.89159867785146085, 'contribution'), (0.89091859185919831, 'priori'),(0.89076537824989432, 'microscope')]
 Most similar word 'repairer' has a cosine similarity of 0.914. 


3. Go back to the `context.py` script, and search for one of the same words, using context-size five. (Use `.most_common(100)` to avoid pages and pages of output.) 
Now do the same for some of the words that the embedding said were nearest. Are their context words similar?

A lot of my words didn't have much context. For example "contexts("rythmic",5)" returned nothing.
"contexts("calculus",5)" returned
a lecherous text-book of the [calculus] or of a reporter's story
but "contexts("repairer",5)" didn't return anything.
We lucked out with 
>>> contexts("dispersion",5)
length to the determination of [dispersion] over the adjacent country in
a loud tone A general [dispersion] of the crowd ensued and
have been expected our complete [dispersion] and the arrest of some
>>> contexts("steadiness",5)
And so hauling with great [steadiness] we brought the monster near
car consequently followed with a [steadiness] so perfect that it would
rather for the quietude and [steadiness] of its population than for
was the suddenness and the [steadiness] of the attack that had
walk with some degree of [steadiness] among temptations and in my
walk with some degree of [steadiness] among temptations and in my

It seems like their context words were a bit similar but not too much. 


4. Go back to the embedding vectors. Now try searching for the sum of two vectors: 
just use `+`! Search for three pairs of words, and copy the 20 nearest neighbors here. 
What do you notice about the single closest vector? How is this different from the previous question?

>>> v1 = get_vector("dispersion")
>>> v2 = get_vector("rhythmic")
>>> v3 = get_vector("calculus")
>>> v11 = get_vector("steadiness")
>>> v22 = get_vector("repellent")
>>> v33 = get_vector("repairer")

>>> nearest(v1+v11, 20)
[(0.98607891518231849, 'dispersion'), (0.98607891518231827, 'steadiness'), (0.95209554381113026, 'inconveniences'), (0.94477513945281932, 'trespassers'), 
 (0.94120615471444136, 'sequestered'), (0.94114937903905582, 'ineffective'), (0.93980199585869961, '_jane'), (0.93965548095856133, 'shifts'), 
 (0.93926265332512227, 'automata'), (0.9371021541843525, 'two-thirds'), (0.93683668399846121, 'disorderly'), (0.93638106027648438, 'maltese'), (0.9355696030597479, 'booty'), 
 (0.93521048045782629, 'renting'), (0.93474258614473382, 'paraphernalia'), (0.93473779947972058, 'personalities'), (0.93420943948421076, 'pretentious'), (0.93394382879986504, 'rustics'),
 (0.93223605336363868, 'approbation'), (0.93168101085982091, 'bestowing')]

>>> nearest(v2+v22, 20)
[(0.98633723908224002, 'rhythmic'), (0.98633723908224002, 'repellent'), (0.95922873725065361, 'candlelight'), (0.95814919049626734, 'mistook'), 
 (0.95623207378032937, 'embraces'), (0.95398769837685604, 'metaphor'), (0.95008623065075748, 'medicinal'), (0.95004733260326002, 'roughness'), (0.94966754929766584, 'lingers'),
 (0.94833360591694049, 'undecipherable'), (0.9478137519644223, 'merging'), (0.94577422438496195, 'gloss'), (0.94436086373034556, 'strands'), (0.94425093715156372, 'turgid'), 
 (0.94399475556653833, 'manageable'), (0.94389725366955635, 'transparency'), (0.94376530654020718, 'banishing'), (0.94353854999053177, 'disagreeably'), (0.94290478434896574, 'symmetry'), 
 (0.94124356477165139, 'child-like')]

>>> nearest(v3+v33, 20)
[(0.97843649322524073, 'repairer'), (0.97843649322524073, 'calculus'), (0.94286672417739525, 'reputations'), (0.91544436385717942, 'berlin'), (0.91327555343087052, "anything'"),
 (0.91032048681341693, 'plagues'), (0.90988343350161949, 'necessities'), (0.90959311462454984, 'inter'), (0.90852475192773796, 'newton'), (0.90598118391568905, 'disciple'), 
 (0.90570391348114865, '179'), (0.90565547159064375, 'validity'), (0.90498246950448258, 'prophets'), (0.9046492257089005, 'incensed'), (0.90453653082941909, 'algebraical'), 
 (0.90414686875560712, 'bono'), (0.90333328589978423, 'locke'), (0.90321458259473375, 'indolent'), (0.90280603340730892, 'epigram'), (0.90239666832878274, '(strange')]

Here the single closest verctor is one of teh two words. This is diffrent than the previous problem in that in the previous example, the most similar vector 
was the word being inputted with a cosine similarity of 1, but here, no word is exactly the same 


5. What about subtraction? Use the same examples as in the previous problem, 
but this time search for the *difference* between vectors (use `-`). 
Comment on what changes and what stays the same.

>>> nearest(v1+v11, 20)
[(0.98607891518231849, 'dispersion'), (0.98607891518231827, 'steadiness'), (0.95209554381113026, 'inconveniences'), (0.94477513945281932, 'trespassers'), 
 (0.94120615471444136, 'sequestered'), (0.94114937903905582, 'ineffective'), (0.93980199585869961, '_jane'), (0.93965548095856133, 'shifts'), (0.93926265332512227, 'automata'), 
 (0.9371021541843525, 'two-thirds'), (0.93683668399846121, 'disorderly'), (0.93638106027648438, 'maltese'), (0.9355696030597479, 'booty'), (0.93521048045782629, 'renting'), 
 (0.93474258614473382, 'paraphernalia'), (0.93473779947972058, 'personalities'), (0.93420943948421076, 'pretentious'), (0.93394382879986504, 'rustics'),
 (0.93223605336363868, 'approbation'), (0.93168101085982091, 'bestowing')]
>>> nearest(v2-v22, 20)
[(0.36523887313186021, 'gentle'), (0.2887523250808447, 'lower'), (0.27761981175209077, 'constant'), (0.24033043104632457, 'high'), (0.22691161094833923, 'valleys'), 
 (0.22569432047943083, 'earth-current'), (0.22259040678929715, 'ears'), (0.22233415287260422, 'alway'), (0.21950528145704964, 'seas'), (0.21940837352240242, 'steady'), 
 (0.21692397948624051, 'millions'), (0.2145133018498169, 'forces'), (0.21425651810642329, 'music'), (0.21030658571618779, 'roar'), (0.21022036299572924, 'regular'), 
 (0.2084514131778693, 'south-west'), (0.20776329152482262, 'hearts'), (0.20611505294818902, 'volcanoes'), (0.20519826920825276, 'higher'), (0.20165896119872664, 'sway')]
>>> nearest(v3-v33, 20)
[(0.48247076845169395, 'vapor'), (0.44859851773914872, 'surfaces'), (0.43699120254941859, 'dense'), (0.43492835747660741, 'funnel'), (0.43295040791122685, 'finely'), 
 (0.43200596686560938, 'hues'), (0.43031867326025686, 'resembling'), (0.42944567008517059, 'rim'), (0.41650125098692975, 'thickly'), (0.41602337925518929, 'grotesquely'), 
 (0.41335588962344988, 'surface'), (0.40631706972141124, 'vellum'), (0.40627059764857643, 'portions'), (0.40585150216634702, 'machinery'), (0.40495213011372228, 'species'),
 (0.4047654670046551, 'spots'), (0.40438733726491083, 'diameter'), (0.40333776419301909, 'dusky'), (0.40294993132451229, 'tapestry'), (0.40213682399400646, 'patterns')]

Here the closest word is in 2/3 cases, not one of the inputted words. This shows that the difference brings out different words when looking at words that are similar.

6. The file `original.vec` is trained in the same way on the same files, but without modifications to the text.
   Load these vectors instead of `horror.vec`. Run the same queries. What do you think is different about how I pre-processed the training data?

>>> v1 = get_vector("whaler")
>>> nearest(v1,20)
[(1.0, 'whaler'), (0.92443566479763317, 'Jane.'), (0.92419430990158813, 'Ellen'), (0.92327861483432772, 'Square.'), (0.92296536776153471, '"Yes."'), 
 (0.92189248601438067, 'Dutee'), (0.92137649953895751, 'Arts.'), (0.92023768155051433, 'Californian'), (0.91957389507680465, 'open."'), (0.91581527991449674, '1845.'), 
 (0.91460277983318083, 'quiet."'), (0.91443629135049009, 'tennis'), (0.91438411186562163, 'gbnewby@pglaf.org'), (0.91403740407287604, 'Introduction'), (0.91388968401852111, 'empty."'), 
 (0.91149684920163876, 'say:--'), (0.91133739640225042, 'Metzengerstein'), (0.9108226638536665, '9.'), (0.90884306730157527, 'Science'), (0.90849904293220063, '1849.')]
>>> v2 = get_vector("Jane")
>>> nearest(v1+v2,20)
[(0.91525226368973767, 'whaler'), (0.91525226368973733, 'Jane'), (0.91346634708449082, 'Ellena'), (0.91111986118723087, '1829,'), (0.90974114118527794, 'Edmund'), 
 (0.90528760100863437, 'Forest_'), (0.90422997693966756, 'Italian_,'), (0.90187079499475786, 'di'), (0.90106406019527485, 'Hamilton'), (0.90040383503568111, '_History'), 
 (0.90036804972859985, 'Bristol'), (0.90033926010263021, 'Edinburgh,'), (0.8952626242116597, '1820'), (0.89510526551588943, 'Year'), (0.89414072259067945, 'Californian'), 
 (0.89375565689858094, '1845.'), (0.8931321837958035, 'Sicilian'), (0.89223092739431242, 'Roche,'), (0.89206496513723343, 'Convent'), (0.89037285562819413, 'Spazac,')]
>>> nearest(v1-v2,20)
[(0.63716510977486185, 'blood-curdling'), (0.63094941497121537, 'stared,'), (0.61693658472444679, 'laughter.'), (0.60814908085432373, 'quiver'), (0.60647374463919612, 'muttering'), 
 (0.60451067674134529, 'sound;'), (0.59828666187361823, 'soft,'), (0.59266199567013966, 'flame;'), (0.59177771953226377, 'scream,'), (0.58992024389350894, 'snapping'), 
 (0.5890295948311477, 'roared'), (0.58871436378220054, 'fury;'), (0.58765698242394793, 'blast'), (0.58522783045018523, 'fluttering'), (0.58482101111619555, 'mocking'),
 (0.58436519292310218, 'dreadful,'), (0.58396947737547455, 'roar,'), (0.582433504865709, 'ghastly,'), (0.5821840120126035, 'breathing.'), (0.58214644008830885, 'twig')]

From looking at these results it seems like in pre-processing the data you took out a lot of things that weren't nouns. For example, names, and years, were removed.


7. The file `ecco_vectors.vec` has embeddings trained by Ryan Heuser on billions of words from 18th century books. 
   Load this file, and try at least five queries. This can include single words or combinations of words. How are these different 
   from the vector results from the smaller Horror fiction data set? What do you notice about words in 18th century books?

>>> v1 = get_vector("bij")
>>> v2 = get_vector("smelt")
>>> v3 = get_vector("horfc")
>>> v4 = get_vector("y'are")
>>> v5 = get_vector("dhat")
>>> nearest(v1, 20)
[(0.99999999999999978, 'bij'), (0.75338503444095128, 'hij'), (0.71925816625206784, 'niet'), (0.70550353507907348, 'zyn'), (0.70504606673076509, 'aan'), 
 (0.69951684100271683, 'geen'), (0.69689517711932125, 'het'), (0.69671450835207727, 'voor'), (0.68038812064478416, 'daar'), (0.67737216317279836, 'iets'),
 (0.66129147174481218, 'zal'), (0.65736363271307163, 'ecn'), (0.64261000004743751, 'heeft'), (0.6408245088661535, 'iemand'),(0.63325909283542159, 'eene'), 
 (0.62667141168738039, 'ik'), (0.62059307923271256, 'als'), (0.62035060542199205, 'een'), (0.60982874839363166, 'kan'), (0.60883247859377543, 'doen')]
>>> nearest(v2, 20)
[(0.99999999999999967, 'smelt'), (0.64800769195946528, 'smell'), (0.63470355347598173, 'finell'), (0.53406565251512705, 'smelling'), (0.51567062242856065, 'stink'), 
 (0.4766016270958997, 'stinking'), (0.46858003374658397, 'smells'), (0.46412781198575348, 'scent'), (0.44064261107636293, 'tafted'), (0.43903468742763546, 'perfume'), 
 (0.42280678166271024, 'odour'), (0.38333113389781187, 'snuff'), (0.38050350004569244, 'rubbed'), (0.37810655826285522, 'flavour'), (0.3764715418839657, 'tailed'), 
 (0.37523677209821465, "boil'd"), (0.37028419362496234, 'stuck'), (0.37021131177735089, 'tobacco'), (0.36524869183977682, 'tafie'), (0.36370077240921417, 'butter')]
>>> nearest(v3, 20)
[(0.99999999999999944, 'horfc'), (0.88064746597278121, 'horse'), (0.87472667565683693, 'horle'), (0.77827323108214752, 'horfe'), (0.67903825723545452, 'horses'), 
(0.62052915221892024, 'steed'), (0.61151097846640301, "horse's"), (0.61120497575965371, 'saddle'), (0.59521688869646239, 'cheval'), (0.59497992302790159, 'courser'), 
(0.55888096451524372, 'cavalry'), (0.54779310226042721, 'cavallo'), (0.5467793046795304, 'mule'), (0.54050414384494871, 'dapple'), (0.53235188999028027, 'bridle'), 
(0.5229795192664799, 'horsemen'), (0.52240552339427948, 'horseback'), (0.5139364421249677, 'foot'), (0.51302901361434161, 'hoife'), (0.51082654788399273, 'coach')]
>>> nearest(v4, 20)
[(0.99999999999999989, "y'are"), (0.68449871648424465, "you're"), (0.58222213904446107, "thou'rt"), (0.53832880150698725, "he's"), (0.53772439267911509, "i'm"), 
(0.53531111880883087, "ihe's"), (0.53450386082650558,"she's"), (0.53092406501226752, "they're"), (0.51923395283595442, "we're"), (0.49946493845258788, "you've"), 
(0.49598944467505734, "you'll"), (0.43006889688005423, "here's"), (0.42820125811557241, "that's"), (0.41967090812068553, "they'll"), (0.41591189014593444, "we've"), 
(0.40789762739588642, 'you'), (0.40679369212912886, 'welcome'), (0.40283904736038462, "she'll"), (0.40268216822471992, 'yourselves'), (0.40200058511917547,'ifab')]
>>> nearest(v5, 20)
[(1.0, 'dhat'), (0.81202794860222705, 'dhe'), (0.76715237148230453, 'yoo'), (0.60163429908784649, 'az'), (0.56597717256207436, 'hoo'), (0.48794892539543605, 'onely'), 
(0.42582231468170689, 'fhal'), (0.4211775013497982, 'dh'), (0.41508291249670504, 'doo'), (0.41301987561698261, 'that'), (0.39533213133965078, 'ithat'), (0.37541520301190068, 'thati'),
(0.36804191118055185, 'th.t'), (0.36583925182204469, 'ihat'), (0.36327425170817162, "t'hat"), (0.35739582996111197, 'tlat'), (0.35500866587143826, 'thai'), (0.35358345879438563, 'lthat'), 
(0.35285207402821983, 'tihat'), (0.35237595737882987, 'hav')]
>>> v11 = get_vector("hij")
>>> v22 = get_vector("smell")
>>> v33 = get_vector("horse")
>>> v44 = get_vector("you're")
>>> v55 = get_vector("dhe")
>>> nearest(v1+v11,20)
[(0.93631859813872964, 'hij'), (0.93631859813872942, 'bij'), (0.75839248862471287, 'het'), (0.75246708385498828, 'zyn'), (0.72669512967275618, 'aan'), (0.72178289835136278, 'geen'), 
 (0.7212100362462095, 'niet'), (0.69935642469614079, 'voor'), (0.69136748889941069, 'iets'), (0.69022882708999367, 'heeft'), (0.68689734114243928, 'daar'), (0.68657184755887724, 'zal'), 
 (0.6769848034560958, 'ecn'), (0.65566148472038932, 'ik'), (0.65288375998329529, 'iemand'), (0.6406942501512789, 'zich'), (0.64017217270816951, 'als'), (0.63541699292692622, 'eene'), 
 (0.63249677208404098, '1k'), (0.63175691810559897, 'doen')]
>>> nearest(v1-v11,20)
[(0.35115165210991711, 'bij'), (0.233518201581146, 'auf'), (0.20947727948766703, 'unb'), (0.20439002182273786, 'etr'), (0.20028330422893287, 'eine'), (0.19898321944874853, 'btr'), 
 (0.19481030050323023, 'ober'), (0.19338894893638586, 'einer'), (0.19249732481870313, 'bei'), (0.19109353710989929, 'zie'), (0.18691086199580143, 'aue'), (0.18534188681260672, 'pos'), 
 (0.18100750565638252, 'sorte'), (0.17912616008887869, 'einem'), (0.17790332354303784, 'bcr'), (0.1769251330133422, 'ert'), (0.17690066770315171, 'ptr'), (0.1721251718318465, 'sotto'), 
 (0.17208364230978851, 'eber'), (0.17199658254731301, 'bic')]
>>> nearest(v2+v22,20)
[(0.9077465758567933, 'smell'), (0.90774657585679319, 'smelt'), (0.86003402487566327, 'finell'), (0.68223262664011508, 'smelling'), (0.65969027978391559, 'scent'), (0.62882686733405002, 'odour'), 
 (0.6224030528267589, 'stink'), (0.59760995357645086, 'smells'), (0.57576997689072384, 'perfume'), (0.55665784344157299, 'stinking'), (0.53211397553588691, 'flavour'), 
 (0.49423766754431814, 'taste'), (0.49249267200128471, 'talte'), (0.49174796751203087, 'tafie'), (0.49025276568887571, 'talle'), (0.48719938173162292, 'tafle'), 
 (0.47429569263151306, 'perfumes'), (0.47209151886230077, 'odoriferous'), (0.47164664939210232, 'tafe'), (0.46872660167481939, 'tafted')]
>>> nearest(v2-v22,20)
[(0.41951895549577634, 'smelt'), (0.23998274532121955, 'overheard'), (0.23070378732889968, 'clapped'), 
 (0.21583970061686425, "knock'd"), (0.2144653309200959, 'knocked'), (0.19619076000984242, 'poke'), (0.19416142930521901, 'dreamt'), (0.19101330922952342, "talk'd"), 
 (0.18909976174051563, 'recollected'), (0.18543895658011719, 'spoke'), (0.18460552331163066, 'pew'), (0.18435772056868843, "leap'd"), (0.18229540814730755, 'coined'), 
 (0.18175869983363757, "figh'd"), (0.18084950238794628, 'ihad'), (0.1799602954489381, 'met'), (0.17925846927458405, 'en'), (0.1790559670351867, 're'), (0.1774566238886216, 'laughed'), 
 (0.17706668504542028, 'slightly')]
>>> nearest(v3+v33,20)
[(0.9697029096513996, 'horse'), (0.96970290965139938, 'horfc'), (0.92168581150428996, 'horle'), (0.78310053691687642, 'horfe'), (0.7367758581291397, 'horses'), 
 (0.68298102887825052, 'steed'), (0.65517565838383074, "horse's"), (0.65435162830513649, 'saddle'), (0.61935548798775319, 'courser'), (0.60165319596492206, 'cheval'), 
 (0.58498688057102388, 'cavalry'), (0.57356014234894026, 'cavallo'), (0.56821469843564998, 'mule'), (0.56443171018641458, 'horsemen'), (0.55833986284122838, 'dapple'), 
 (0.55604114489276069, 'bridle'), (0.54931021087733067, 'horseback'), (0.54461997251698424, 'mounted'), (0.54046971499401786, 'hoife'), (0.53227181625331743, 'steeds')]
>>> nearest(v3-v33,20)
[(0.27106265313287664, 'onc'), (0.26598062333051331, "l't"), (0.25710733445093792, "l'r"), (0.25362276889334778, ':e'), (0.24428726330615219, 'horfc'), (0.24124446415897266, 'lf'),
 (0.23755628983700416, 'therc'), (0.23673130600809728, 'df'), (0.23367624536309184, 'rcn'), (0.23199525138805829, 'thlie'), (0.23106219483621088, "l'o"), (0.23101602842080304, 'crs'),
 (0.22854482506988633, 'lre'), (0.22843777189520173, 'rai'), (0.227865417910114, 'i:e'), (0.22764480992963743, 'foie'), (0.22739221381784261, 'dll'), (0.22499825278243765, 'liet'), 
 (0.22235146385651341, 'rth'), (0.22217741899450216, "l'i")]
>>> nearest(v4+v44,20)
[(0.91774144411273184, "y'are"), (0.91774144411273173, "you're"), (0.71801753547013214, "i'm"), (0.70133839248242069, "he's"), (0.68842261123867243, "thou'rt"), (0.68152921167797376, "ihe's"),
 (0.66571022643674982,"she's"), (0.65724423040813029, "they're"), (0.63960496074758955, "we're"), (0.61541967859507118, "you'll"), (0.59056529659648871, "you've"), (0.55089523020549236, "that's"), 
 (0.50827383857375763, 'you'), (0.50599092193946937, "they'll"), (0.50386534004930361, "here's"), (0.49026400281864257, "you'd"), (0.48784415591597874, "she'll"), (0.48432373362595627, "there's"), 
 (0.48341857527101273, 'yes'), (0.4795796844832021, "he'll")]
>>> nearest(v4-v44,20)
[(0.39717835006188057, "y'are"), (0.25293470910829086, 'camillo'), (0.2036795160738083, 'verona'), (0.20153482443395992, 'wol'), (0.19808480528569891, 'husbandry'), (0.18969140577156487, 'exe'), 
 (0.17756250210167041, 'grete'), (0.17704923918788879, 'haue'), (0.17669512053259773, 'venereal'), (0.1718745933188327, 'dili'), (0.16833095012711802, 'lep'), (0.16731569813712302, 'clo'), 
 (0.16434642776779701, 'oc'), (0.1613750062224073, 'lucio'), (0.16112589341025979, 'felfe'), (0.1605122724496145, 'turpe'), (0.15645890516282637, 'angelo'), (0.1560437546612562, 'whiche'), 
 (0.15432418989756463, 'ploy'), (0.1540424690233379, 'faced')]
>>> nearest(v5+v55,20)
[(0.95184766339005822, 'dhat'), (0.95184766339005811, 'dhe'), (0.78728783460168406, 'yoo'), (0.650150929412875, 'az'), (0.61700324172616949, 'hoo'), (0.48860706511701235, 'onely'), 
 (0.47355866029044735, 'dh'), (0.46732103385794177, 'doo'), (0.43691163308551173, 'fhal'), (0.40116232146758224, 'th.e'), (0.38721351212428817, 'hav'), (0.3831796264075481, 'iz'), 
 (0.37623172095502416, 'ov'), (0.35857050883165209, 'tthe'), (0.35786659571124818, 'ftil'), (0.35557045823631206, 'ithat'), (0.35524587373508049, 'th:e'), (0.35376827862827387, 't!e'), 
 (0.34776230569964051, 'ihal'), (0.34240981020577865, 'ithe')]
>>> nearest(v5-v55,20)
[(0.30657140391577042, 'dhat'), (0.29347912049592456, 'that'), (0.19793262065136014, 'tlat'), (0.19727061555126441, 'goodness'), (0.19725245687904891, 'thac'), 
 (0.18803766005052999, 'thiat'), (0.18634009302014526, 'tlhat'), (0.18554640377519818, 'ithat'), (0.18403250409859317, 'tliat'), (0.18348337834983541, 'tllat'), (0.18341418225414868, "t'hat"), 
 (0.18324460941455167, 'thlat'), (0.18104881749845908, 'rectitude'), (0.17950676446900868, 'thati'), (0.1781255073668172, 'th.t'), (0.17726536471767554, 'verily'), 
 (0.17679202689977125, 'existence'), (0.17546008953458453, 'tihat'), (0.17201762066669557, 'lhat'), (0.17184874503929409, 'lthat')]

From looking at these queries I made (I looked in word_ids to find some words I liked) it seems like there are a lot more mispelled or foreign words that the horror novels. This 
might be a result of the corpus not being parsed as nicely in some cases, but it was interesting to see (since it was 18th century) how words were different back then.

"""

import numpy as np
import sys

reader = open(sys.argv[1], encoding="utf-8")

# The first line of the file gives the dimensions of the matrix
first_line = reader.readline()
fields = first_line.split(" ")
n_rows = int(fields[0])
n_cols = int(fields[1])

word_vectors = np.zeros((n_rows, n_cols))
vocabulary = []
word_ids = {}

for line in reader:
    fields = line.strip().split(" ")
    word = fields[0]
    vector = np.array(fields[1:], dtype=float)
    norm = np.linalg.norm(vector)
    
    # in some odd cases a word may have no vector
    if norm == 0.0:
        continue
    
    # record the word and its row position in the matrix
    word_id = len(vocabulary)
    vocabulary.append(word)
    word_ids[word] = word_id
    
    word_vectors[word_id] = vector / norm

## look up the id for a word, and grab the row associated with that word
def get_vector(word):
    return word_vectors[ word_ids[word] ]

## define a shortcut
v = get_vector

def nearest(v, n):
    norm = np.linalg.norm(v)
    if norm > 0.0:
        v /= norm
    return(sorted(zip(word_vectors.dot(v), vocabulary), reverse=True)[:n])
