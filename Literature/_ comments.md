- Undersøg mass maximation
- Ift. [3.], undersøg hvad der har størst indflydelse på usikkerheden i prediction ud fra modellen. Hvilke arbejdere/maskiner er bottleneck/devils
- Overordnet: fordeling og hvad der påvirker den/model + prediction/ekstrapolering
- Modellering som opstart til flow i nogle tanke? Eller det er jobs som i eksisterende artikler, hvor processen skal køre færdig får output kan bruges? Forstå e.g. hoved-reaktoren, som starter op, når opstart leverer den (konstant) flow i et stykke tid T (stokastisk) indtil muligvis rens, ny påfyldning, opstart etc.
- Gør det tilpas generelt (hvis muligt) så kan bruges på andet data og indentificere bottlenecks
- Kig lidt mere ind i TPM (5.) og se om der er noget at hente i forhold til modellering af adfærd på arbejdspladserne.
- Læs Rita et al. for appraoches til optimering af OEE

6,7,8 omhandler fit af kontinuert PH til data

# Optimering
- Muligt at tilvælge daily maintenance (preventive). Hvad giver det af værdi?

# 1. A Proposal for Production Scheduling Optimization Method with Worker Assignment Considering Operation Time Uncertainty
- Små modeller de løser. Kan evt. bruge DTU licenseret Gurobi software til at forbedre tiderne.
- Burde ikke være så svært at udvide deres model til at passe på et sekventielt setup, eller evt. sekventielt setup med flere maskiner i parallel
- Skal nærlæse modellen lidt mere for at forstå 100%, men ligner de har lavet lidt simplificeringer, og ikke brugt "rigtig" stokastisk OR (som er lidt tungt at køre, eller skal køres mange gange).


# 2. Analysis of assembly-time performance (ATP) in manufacturing operations with collaborative robots: a systems approach
- ATP: lyder som et godt sted at starte med undersøgelse og berebgninger. Sætte systemet op anaylitisk og undersøg. Undersøg med lager mulighed!!
- bottleneck identification and mitigation
- mange referencer til papers der analyserer robot-menneske systemer og kigger på forskellige performance indexes
- "The job schedules, task assignment and robots allocation problems in reconfigurable assembly lines with collaborated human operators and mobile robots are formulated and solved through a hybrid optimisation method by Maganha et al. (2019)."
- "Recently, Zhang, Huang et al. (2021) develop a method to evaluate flow time in a collaborative assembly systems using phase-type distributions to describe both human and cobot preparation processes and their joint assembly process, where each task time is described by an exponential distribution."
- viser en måde med 1 overgang mellem processer. vil gerne have flere (adapt beviser)
- Kan bruge dette til opstart af processen. Tænker også at kigge på efter opstart, hvad er da bedst. Evt. en samlet opstart, kørsel model. Hvis process bliver startet og lukket mange gange (rens af tanke e.g.), kan dette være meget godt.

- on-line monitoring and control of human– robot interactions and dynamic allocation or reassignment of tasks, which are of significant importance to optimise system performance and ensure safety in real-time
- analysing and accommodating correlations between multiple activities and the associated task times in the model, and seeking analytical derivations under certain conditions in addition to numerical solutions.
- study multiple stage assembly systems, where finite buffers are used to connect different stages, and blockage and starvation introduced due to assembly time variations can propagate up- and downstream to impact system throughput.


# 3.
- Netværk af nodes, hvor hver node er en akkumulering af G/G/m køer. Bruger "overtaking" fordeling (afhængig af hvor i systemet og load), da FCFS (first-come-first-serve) ikke gælder. Kan være god, da håndterer arbitrært netværk
- har algoritme for at regne på EPT og overtaking fordelinger ud fra data (in-time, out-time)
- Fine tider ift. computing/simulering og fitting. De bruger C#, så bør overveje om den del skal laves i C# eller om der findes tilsvarende python biblioteker, der kan matche den hastighed

# 4.
Machine learning til at identificere features der påvirker Cycle-time og Work-time. Derefter en classifier, der forduser niveau a cycle-time og work-time. opnår 73% accuracy (ok). Kan måske bruges til at identificere hvad der er vigtigst for os at modellere for bedre statistisk model?


# 6
Bruger sojourn times, så kan ikke direkte bruges, hvis man bare vil "klaske" en phase type på de data der eksisterer. Kan dog nok drejes, så kun bruger målte tider til færdiggørelse i hvert step, så hvert step ses som en markov process, men hvor vi kun målet in og output
- Kan påtvinge struktur i generator matrix, ved at sætte elementer til nul (evt. reset nul efter et par trin for numeriske årsager)



#
modellere deviations og ufordusete nedbrud. 

m og uden ufordusete begivenheder

effekter downstream. hvornår skal man advare upstream