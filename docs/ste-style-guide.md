# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

This section gives the technical names and the technical verbs of crashinsight. The README uses each term with only this meaning.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **GES** | The NHTSA General Estimates System sample of police-reported crashes | the database, the dataset (alone) |
| **crash** | One police-reported crash, identified by `CASENUM` | accident (except the `ACCIDENT` file), case, incident |
| **vehicle row** | One row of `VEHICLE.CSV`, identified by `CASENUM` and `VEH_NO` | record, sample, observation |
| **codebook** | The JSON file with the labels and the sentinel codes of each variable | dictionary, mapping, lookup table |
| **code** | The integer value of a variable in a GES file | value (for a raw code), number |
| **label** | The text that the codebook gives to a code | name, description, category text |
| **sentinel code** | A code that means unknown, not reported or not applicable | missing value (for the raw code), placeholder |
| **KABCO** | The police injury scale: K fatal, A serious, B minor, C possible, O no apparent injury | severity score, injury class |
| **severity** | The KABCO letter of the most severe injury in the vehicle (`MAX_VSEV`) | outcome (alone), damage |
| **KA** | The binary target: 1 if the severity is K or A, else 0 | severe, serious crash, fatal flag |
| **feature** | One model input column, made from one or more GES variables | predictor, attribute, field |
| **feature spec** | The lists of numeric, binary and nominal features (`FeatureSpec`) | schema (for features), column list |
| **leakage column** | A column that describes the outcome or the events after the crash | target proxy, post-crash feature |
| **weight** | The GES case weight, `WEIGHT` | sampling factor, multiplier |
| **stratum** | A variance stratum, `PSUSTRAT` | layer, block |
| **PSU** | A primary sampling unit, the cluster of the design | cluster (alone), site |
| **weighted rate** | The KA share that uses the weight | national rate, adjusted rate |
| **domain** | A subpopulation, for example one body group, for a weighted rate | subgroup, segment |
| **fold** | One part of the crash-grouped cross-validation | split (for one part), partition |
| **hold-out test** | The crashes that the final model never sees in training | validation set, test fold |
| **baseline** | The majority-class model (`majority`) | dummy, naive model |
| **pipeline** | The scikit-learn `Pipeline` of preprocessor and estimator | workflow, chain |
| **odds ratio** | The exponent of the `newer` coefficient in the age regression | risk ratio, effect size |
| **model card** | The Markdown file that describes one saved model, its metrics and its limits | model report, datasheet |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **decode** | Change a code into a label, a number or NaN with the codebook |
| **validate** | Check the columns and the codes of a file against the schema and the codebook |
| **build** | Make the analysis table from the GES files |
| **join** | Add ACCIDENT and PERSON columns to each vehicle row |
| **weight** | Use the GES weight in a rate or in a regression |
| **train** | Fit a pipeline on the training crashes |
| **evaluate** | Measure a pipeline on rows that it did not see in training |
| **resample** | Draw crashes or PSUs with replacement for a bootstrap |
| **explain** | Measure the drop in KA PR-AUC when one feature is shuffled |
