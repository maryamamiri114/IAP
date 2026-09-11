# Annotation Codebook: Intent Quality Evaluation

Guidelines for human evaluation of inferred user intents in personalized question answering.

**Annotators:** 2, working independently
**Scale:** 1 to 5 Likert, four criteria


## 1. What this study is about

Community question-answering sites contain questions where the same wording can come from people with very different underlying goals. Someone asking *"Should I change my job?"* might want a decision framework, or help diagnosing why they're unhappy, or permission to leave. A response that serves one of these people may not serve the other.

We are evaluating a system that, before answering, tries to infer the asker's **underlying motivations**, which we call their *intents*. Your task is to judge the quality of those inferred intents.

**You are not evaluating the answer.** You will not see it. You are judging only whether the system correctly and usefully identified what the person was actually after.

---

## 2. What you will see

Each screen shows three things:

1. **The question**, as posted to the community site.
2. **The narrative**, the asker's own description of their situation and why they're asking.
3. **The inferred intent set**, a numbered list produced by the system.

You will rate the intent set on **four criteria**, each on a **1 to 5 scale**.

All examples in this codebook use the same instance, so that you can see how the four criteria differ from one another when applied to identical material.

---

## 3. General principles

**Rate the set as a whole, not each intent separately.** One score per criterion per instance, regardless of how many intents are listed.

**Ignore writing quality.** Grammar, phrasing, and formatting are not being evaluated. A clumsily worded intent that identifies the right motivation should score well.

**Don't reward length.** A set of six intents is not automatically more complete than a set of two. What matters is coverage of what's actually there.

**Judge against the narrative, not against your own opinion.** If the system identifies a motivation the asker plausibly has, based on what they wrote, it is correct, even if you personally would have emphasized something else.

**Use the whole scale.** If everything looks like a 4 or 5, re-read the anchors. Scores of 2 and 3 exist and should be used when warranted.

**When genuinely torn between two adjacent scores**, pick the lower one. Systematic optimism inflates results more than systematic pessimism deflates them.

**Do not discuss specific instances with the other annotator** during the main round.

---

## 4. The four criteria

Rate in the order given. Each criterion is independent. A set can score 5 on one and 2 on another.

---

### 4.1 Faithfulness

> **Is every stated intent supported by the narrative and relevant to the question, with nothing invented?**

This is a check for **fabrication**. You are looking for intents that assert motivations the asker never expressed and that don't follow from what they wrote.

**Ask yourself:** For each listed intent, can I point to the part of the narrative that supports it? If not, is it at least a reasonable reading of what's there?

**This is not:** whether the intents are *complete* (that's 4.2), or whether they're *insightful* (that's 4.3 and 4.4). A single obvious intent, fully supported, is a 5 on Faithfulness.

| Score | Definition |
|:---:|---|
| **5** | Every intent is supported by the narrative; nothing invented. |
| **4** | All supported; at most one minor unsupported detail. |
| **3** | Most supported; one clearly unsupported intent. |
| **2** | Several invented or unsupported intents. |
| **1** | Largely fabricated, or contradicts the narrative. |

#### Examples

> **Question:** Should I tell my manager I'm considering leaving?
>
> **Narrative:** I've been at my company three years and recently started interviewing elsewhere. Nothing is finalized. My manager has been supportive of me and I feel guilty keeping this from her, but I've also heard stories of people getting sidelined once they mention they're looking. My performance review is in six weeks.

**Score 5**

> 1. Wants to weigh the professional risk of disclosing against their discomfort with concealment.
> 2. Wants to preserve a relationship with a manager they describe as supportive.
> 3. Concerned about how the performance review interacts with the timing.

Every element traces to the narrative: the risk and guilt tension, the supportive manager, the six-week review.

**Score 4**

> 1. Wants to weigh disclosure risk against their discomfort with concealment.
> 2. Wants to preserve a relationship with a supportive manager.
> 3. **Wants to know how colleagues will perceive them if word spreads.**

The first two are solid. The third isn't in the narrative, but it's a small, plausible extension rather than an invention. One minor unsupported detail.

**Score 3**

> 1. Wants to weigh disclosure risk against concealment.
> 2. **Is dissatisfied with their compensation and wants to negotiate a raise.**
> 3. Concerned about the timing of the review.

Compensation is never mentioned and isn't implied anywhere. One clearly unsupported intent among otherwise sound ones.

**Score 2**

> 1. **Wants to leave because of a conflict with their manager.**
> 2. **Is seeking a promotion at their current company.**
> 3. **Wants advice on salary negotiation.**

Multiple inventions. The manager conflict actively misreads the narrative, which describes her as supportive.

**Score 1**

> 1. **The asker has already accepted another offer and wants to know how to resign professionally.**
> 2. **They want to negotiate a counteroffer.**

Directly contradicts the narrative, which states nothing is finalized.

---

### 4.2 Completeness

> **Does the intent set capture the essential motivations present, without omitting important ones?**

This is the **coverage** counterpart to Faithfulness. A set can be perfectly accurate and still miss most of what the asker wants.

**Ask yourself:** Re-read the narrative and list the motivations you can identify. How many appear in the intent set? Are any *important* ones missing?

**This is not:** a count. Three intents that cover everything score higher than six that cover the same ground redundantly.

**Important versus minor:** A motivation is *important* if a response ignoring it would leave the asker unsatisfied. It's *minor* if it's a detail they'd accept being skipped.

| Score | Definition |
|:---:|---|
| **5** | Captures all essential motivations present. |
| **4** | Captures all major motivations; misses something minor. |
| **3** | Captures the primary motivation; misses a significant secondary one. |
| **2** | Captures only a fragment of what is present. |
| **1** | Misses the central motivation entirely. |

#### Examples

> **Question:** Should I tell my manager I'm considering leaving?
>
> **Narrative:** I've been at my company three years and recently started interviewing elsewhere. Nothing is finalized. My manager has been supportive of me and I feel guilty keeping this from her, but I've also heard stories of people getting sidelined once they mention they're looking. My performance review is in six weeks.

*Motivations present in this narrative: the risk of disclosure weighed against the discomfort of concealment; the wish to preserve a valued relationship; the interaction with the upcoming review.*

**Score 5**

> 1. Wants to weigh disclosure risk against discomfort with concealment.
> 2. Wants to preserve the relationship with a supportive manager.
> 3. Wants to understand how the review timing affects the decision.

All three threads in the narrative are represented.

**Score 4**

> 1. Wants to weigh disclosure risk against discomfort with concealment.
> 2. Wants to preserve the relationship with a supportive manager.

Both major motivations covered. The review-timing concern is omitted, real but secondary to the central dilemma.

**Score 3**

> 1. Wants to know whether disclosing carries professional risk.

Captures the primary concern, but the relational dimension, guilt toward a manager they respect, is a significant part of why they're asking, and it's absent.

**Score 2**

> 1. Wants to know whether to tell their manager.

A fragment. It restates the decision without capturing any of what makes it difficult for this person.

**Score 1**

> 1. Wants advice on how to interview successfully while employed.

The central motivation, the disclosure dilemma, is missing entirely.

---

### 4.3 Motivational fidelity

> **Do the intents articulate *why* the person is asking, rather than restating *what* they asked?**

The question tells you *what*. A useful intent tells you *why*: the goal, worry, or need sitting behind it.

**Ask yourself:** If I deleted the question and read only the intents, would I learn something about this person's situation? Or would I just be reading the question again in different words?

**A quick test:** Could this intent be attached to almost anyone asking this question? If yes, it's probably restatement.

| Score | Definition |
|:---:|---|
| **5** | Clearly articulates *why*, in terms of the user's goals. |
| **4** | Mostly *why*, with some restatement of *what*. |
| **3** | An even mix of motivation and restatement. |
| **2** | Largely restates the question. |
| **1** | Pure paraphrase of the question. |

#### Examples

> **Question:** Should I tell my manager I'm considering leaving?
>
> **Narrative:** I've been at my company three years and recently started interviewing elsewhere. Nothing is finalized. My manager has been supportive of me and I feel guilty keeping this from her, but I've also heard stories of people getting sidelined once they mention they're looking. My performance review is in six weeks.

**Score 5**

> 1. Wants to resolve a tension between loyalty to someone who has supported them and the need to protect their own position.
> 2. Is looking for a way to act with integrity without accepting unnecessary professional cost.

These describe what's at stake for this person. Neither could be read off the question alone.

**Score 4**

> 1. Wants to understand whether disclosure is professionally risky.
> 2. Wants to act with integrity toward a manager they respect.

The second is genuine motivation. The first is closer to a restatement of the question's subject matter.

**Score 3**

> 1. Wants to know whether to tell their manager they are looking.
> 2. Is worried about being treated differently afterward.

One restatement, one motivation. An even mix.

**Score 2**

> 1. The user wants to know if they should tell their manager they are considering leaving.
> 2. The user wants advice on this situation.

Almost entirely the question restated, plus a contentless second item.

**Score 1**

> 1. Should the asker tell their manager they are considering leaving?

The question with the words rearranged.

---

### 4.4 Beyond-narrative contribution

> **Do the intents make explicit something the narrative leaves implicit, rather than paraphrasing what's already written?**

This is the most demanding criterion. The narrative already says a lot. The question here is whether the system **inferred** something or merely **located** it.

**Ask yourself:** Is there anything in this intent set that the asker did not say, but that follows from what they said?

**Important:** an inference must still be *justified*. Something invented is not a contribution; that's a Faithfulness failure. The best scores go to intents that are both unstated and well-grounded.

**Expect variation.** Some narratives spell out the motivation almost completely, leaving little room to infer. A low score here may reflect the instance, not the system. Rate what's in front of you; the analysis accounts for this.

| Score | Definition |
|:---:|---|
| **5** | Articulates a motivation not stated; the inference is well justified. |
| **4** | Some genuine inference beyond the stated text. |
| **3** | Mostly restates, with slight extension. |
| **2** | Near-verbatim restatement of the narrative. |
| **1** | Copies the narrative with no inference. |

#### Examples

> **Question:** Should I tell my manager I'm considering leaving?
>
> **Narrative:** I've been at my company three years and recently started interviewing elsewhere. Nothing is finalized. My manager has been supportive of me and I feel guilty keeping this from her, but I've also heard stories of people getting sidelined once they mention they're looking. My performance review is in six weeks.

*Stated in this narrative: the guilt, the sidelining stories, the review in six weeks. Anything beyond these is inference.*

**Score 5**

> 1. The mention of the upcoming review suggests the asker is trying to control the *sequence* of events, securing a rating they've earned before introducing information that might affect it.
> 2. Their guilt suggests they are seeking not just a tactical answer but reassurance that withholding this is defensible.

Neither is stated. Both follow from details the asker included, the six-week review and the word "guilty," and both change what a good answer would address.

**Score 4**

> 1. Wants reassurance that not disclosing is ethically acceptable in this situation.
> 2. Wants to weigh disclosure risk against concealment discomfort.

The first is a genuine inference: the asker mentions guilt but never asks for reassurance. The second is stated.

**Score 3**

> 1. Wants to avoid being sidelined at work.
> 2. Feels conflicted about the decision.

"Sidelined" appears in the narrative; framing it as a want is a slight extension. The second adds nothing.

**Score 2**

> 1. Feels guilty about keeping this from their manager.
> 2. Has heard that people get sidelined after mentioning they're looking.

Both are the narrative's own sentences, lightly reworded.

**Score 1**

> 1. The asker has been at the company three years and recently started interviewing elsewhere. Nothing is finalized.

The narrative, copied.

---

## 5. Hard cases

**The set repeats itself, three intents that are really one.**
Score Completeness on distinct motivations covered, not items listed. Three paraphrases of one motivation count as one motivation.

**One intent is excellent, another is fabricated.**
Faithfulness drops, since a fabrication is present. Motivational fidelity and Beyond-narrative may stay high if the good intent carries them. That's fine, as the criteria are independent.

**The narrative is vague and barely explains anything.**
Then there's less to be faithful to and less to omit. Judge against what's actually there. If the narrative supports only one motivation and the system found it, Completeness is a 5.

**I'd have framed it differently, but the system's reading is defensible.**
Score it as correct. You're judging plausibility, not matching your own interpretation.

**The intent list is very long, eight or nine items.**
Length alone is neither good nor bad. Long lists often drag Faithfulness down, since there are more chances to invent, while pushing Completeness up. Score each criterion on its own terms.

**An intent is true but useless, for example "the user wants a helpful answer."**
Faithful, but contributes nothing. It shouldn't raise Completeness, and it should pull down Motivational fidelity if it's a large share of the set.

**I can't tell whether something is inference or restatement.**
Re-read the narrative and look for the specific sentence. If you can point to it, it's restatement. If you're reconstructing it from several details, it's inference.

---

## 6. Quality checks

A small number of instances in your batch have been seeded with an intent set taken from a **different question**. These are topically plausible but do not match the narrative in front of you. They should receive a Faithfulness score of **2 or below**.

These exist to verify careful reading. If you find yourself rating quickly and not consulting the narrative, slow down.

---

## 7. Recording ratings

For each instance, record:

| Field | Values |
|---|---|
| Instance ID | (given) |
| Faithfulness | 1 to 5 |
| Completeness | 1 to 5 |
| Motivational fidelity | 1 to 5 |
| Beyond-narrative contribution | 1 to 5 |
| Notes | optional, free text |


