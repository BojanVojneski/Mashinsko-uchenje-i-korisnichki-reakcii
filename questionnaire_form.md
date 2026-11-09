# Questionnaire: Ready-to-Publish Package

Language: **English** (the comments are English, and you are recruiting from international communities).
Replace every `[PLACEHOLDER]` before publishing.

---

## 0. What you need first
1. Step B1 done: `data/processed/comments.csv` (one row per comment, with the 28 rater-mean emotion columns, `soglasnost`, `n_rater`).
2. Run `python make_sample.py`. It writes the 100-comment sample (10 blocks of 10, 2-3 comments per emotion group in each block).
3. **Read all 100 comments yourself.** The script flags links, mentions, and usernames, but you must check for anything personal, hateful, or upsetting beyond what you are comfortable posting. To drop a comment: `python make_sample.py --exclude <id>,<id>`.
4. Teacher/school OK for the questionnaire and for posting in public communities (plan section 11, P4).

---

## 1. Route A: build all forms automatically (recommended)
Files: `build_forms.gs` and `panel_forms_input.csv` (from step 0).
1. Upload `panel_forms_input.csv` to your Google Drive.
2. Open script.google.com, create a project, paste `build_forms.gs`, and fill in `contact` and `school`.
3. **Test first:** leave `testBlockOnly: 1`, run `buildForms()`, approve the permissions, open the form from the log and check every page, on a phone too.
4. Delete the test form and the "Panel responses" test spreadsheet, set `testBlockOnly: null`, and run again.
5. The new spreadsheet "Panel responses" has a `links` sheet with each block's share link and the **correct attention-check answer**. Responses from all forms arrive in the same spreadsheet, one tab per form.

The script was written without being run in Google's environment. If it errors, use Route B for that form.

---

## 2. Route B: build one form by hand (same content)

**Form title:** `Emotion in online comments (part N of 10)`

**Form description:**
> Hi! I'm a high school student doing a research project on how people perceive emotion in short online comments. You will read 11 short Reddit comments and say how the writer seems to feel. No special knowledge is needed and there are no right or wrong answers: I'm interested in your honest first impression. Each part of this survey uses different comments.

### Page 1: Consent
**Text block:**
> **About this survey**
> **Purpose:** a school research project about how people perceive emotion in short online comments.
> **What you will do:** read 11 short, real Reddit comments and choose how the writer seems to feel. It takes a few minutes.
> **Voluntary:** you can stop at any time by closing the page. Your answers are saved only if you press Submit.
> **Anonymous:** I do not ask for your name or email. Answers are used only for this project and reported in summary, never individually.
> **Content note:** the comments are real and some contain strong or rude language.
> **Contact:** `[YOUR CONTACT EMAIL OR TEACHER NAME]` (`[YOUR SCHOOL]`).
> You must be 18 or older to take part.

**Two separate required Checkboxes questions** (each with a single option):
1. Title `Eligibility`: option `I am 18 or older.`
2. Title `Consent`: option `I have read the information above, I understand it, and I agree to take part.`

### Page 2: Instructions + comments
**Section title:** `How to answer`
> For each comment, choose how the WRITER seems to feel (not how you feel about it). Go with your first impression, answer on your own, and please do not look anything up.
> **Positive:** e.g. joy, gratitude, admiration, amusement
> **Negative:** e.g. anger, sadness, disgust, disappointment
> **Ambiguous:** feelings that can go either way depending on context, e.g. surprise, curiosity, confusion, realization
> **Neutral:** no clear emotion; factual or informational
> **I can't tell:** the comment is too unclear to judge
> The second question under each comment is optional: tick up to 3 emotions that you see.

**For each of the 10 comments (+ the attention check as the 6th item), add three things in this order:**
1. **Title and description block:** title `Comment N of 11`, description = the comment text.
2. **Multiple choice (required):** title `[CODE] How does the writer of this comment seem to feel?`, options: `Positive`, `Negative`, `Ambiguous`, `Neutral`, `I can't tell`.
3. **Checkboxes (optional):** title `[CODE] Which emotions do you see? (optional, tick up to 3)`, description `Listed in this order: positive, negative, ambiguous, neutral.`, response validation "Select at most 3" with message `Please tick at most 3.`, options in this order (no commas):
   * admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief
   * anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness
   * confusion, curiosity, realization, surprise
   * neutral

`CODE` comes from `panel_sample.csv` (for example `B03-C07`). For the attention check use `B03-ATT`.

**Order:** Google Forms can't shuffle the comments without separating each comment's two questions, so keep the order from the file (it was already shuffled with a fixed seed).

### Page 3: About you (all optional)
* Short answer: `What is your first language?`
* Multiple choice: `How would you rate your English?` with `1 - Basic`, `2 - Intermediate`, `3 - Good`, `4 - Very good`, `5 - Native or near-native`
* Multiple choice: `Where did you find this survey?` with `Friends or school`, `Reddit`, `Discord`, `Other`

### Settings
* Collect email addresses: **off**. Limit to 1 response: **off** (it forces a Google sign-in). Allow response editing: off. Progress bar: on.
* Responses tab: link to a spreadsheet.
* **Confirmation message:**
  > Thank you very much for helping with my project! Your answers have been recorded. If you would like to help more, other parts of this survey (with different comments) are available wherever you found this link.

---

## 3. Attention checks (one per block, rotating)
| Block | Comment | Correct answer |
|---|---|---|
| 1, 4, 7, 10 | Thank you so much, this is the best news I have had all week!! | Positive |
| 2, 5, 8 | This is disgusting and I am furious that they would do this. | Negative |
| 3, 6, 9 | The meeting is at 3 pm in room 12. | Neutral |

**Rule:** exclude any respondent who gets their block's check wrong. Record how many.

---

## 4. Recruitment post (Reddit / forums)
**Title:** `[Academic] High school research project: how do people read emotion in short online comments? (anonymous, 18+, short)`

**Body:**
> Hi everyone! I'm a high school student working on a research project about how people perceive emotion in short online comments, and I'm collecting ratings from volunteers.
>
> - **What:** read 11 short Reddit comments and pick how the writer seems to feel
> - **Who:** anyone 18 or older
> - **Anonymous:** no name, email, or login
> - **Content note:** real comments, some contain strong language
> - **Link:** `[FORM LINK]`
>
> Please answer on your own and go with your first impression. There are other parts with different comments, so feel free to do more than one. I'm happy to share a summary of the results here once the project is done. Thank you so much!

**Short version (Discord):**
> Hi! High school research project on how people read emotion in short online comments: 11 Reddit comments, anonymous, 18+, a few minutes. `[FORM LINK]` Thanks a lot! (Content note: some comments contain strong language.)

**Message to moderators (send before posting):**
> Hi! I'm a high school student doing a research project on perceived emotion in online comments. Would it be OK to post a short anonymous survey (18+, no personal data) in your community? I'm happy to follow any format rules you have. Thank you!

**Posting rules for yourself**
* Don't mention the model or your hypotheses.
* Read each community's rules first; ask before posting in Discord servers; post once per community.
* Spread each block across several communities (not one community per block).
* Make no promises in the post that you won't keep (for example sharing results).

---

## 5. Before you publish: checklist
- [ ] Teacher/school OK obtained
- [ ] All 100 comments read by you; problem comments removed with `--exclude`
- [ ] One form tested end to end on a phone; a few people piloted it (their answers are excluded from the analysis)
- [ ] Contact and school filled in on the consent page
- [ ] Attention-check answers saved (the `links` sheet has them)
- [ ] `panel_sample.csv` (full file with original groups) kept out of the forms and out of any public repo
- [ ] Tracking sheet ready (below)

## 6. Tracking sheet (one row per post)
| Block | Community | Date posted | Valid responses so far |
|---|---|---|---|
| | | | |

Stop a block when it has about 8 valid raters (minimum 5 per comment). Don't edit questions or comments after the first real response; if you must, record it in `docs/decisions.md`.
