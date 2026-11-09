/**
 * Builds the questionnaire forms (one per block of 10 comments) from panel_forms_input.csv.
 *
 * HOW TO USE
 * 1) Upload panel_forms_input.csv (columns: block, code, text) to your Google Drive.
 * 2) Go to script.google.com > New project, paste this whole file, and edit CONFIG below.
 * 3) TEST FIRST: set testBlockOnly to 1, run buildForms(), approve the permissions, and open
 *    the form it creates. Check every page by hand (on a phone too).
 * 4) Set testBlockOnly to null and run buildForms() again to build all blocks. (Delete the
 *    test form and test spreadsheet from Drive first, so you don't end up with duplicates.)
 * 5) Links appear in the log and in the "links" sheet of a new spreadsheet "Panel responses".
 *    Responses from every form go to that same spreadsheet, one tab per form.
 *
 * NOTE: this script was written without being run in Google's environment. If a line fails,
 * the error message says which one; the manual route in questionnaire_form.md gives the same form.
 */

const CONFIG = {
  csvName: 'panel_forms_input.csv',
  totalBlocks: 10,
  testBlockOnly: 1,   // 1 = build only block 1 for testing; null = build all blocks
  contact: '[YOUR CONTACT EMAIL OR TEACHER NAME]',
  school: '[YOUR SCHOOL]'
};

const GROUP_CHOICES = ['Positive', 'Negative', 'Ambiguous', 'Neutral', "I can't tell"];

// Listed in group order: positive, negative, ambiguous, neutral (no commas in labels!)
const EMOTIONS = [
  'admiration', 'amusement', 'approval', 'caring', 'desire', 'excitement', 'gratitude',
  'joy', 'love', 'optimism', 'pride', 'relief',
  'anger', 'annoyance', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
  'fear', 'grief', 'nervousness', 'remorse', 'sadness',
  'confusion', 'curiosity', 'realization', 'surprise',
  'neutral'
];

// One obvious comment per block, rotating. Exclude respondents who get it wrong.
const ATTENTION = [
  'Thank you so much, this is the best news I have had all week!!',
  'This is disgusting and I am furious that they would do this.',
  'The meeting is at 3 pm in room 12.'
];
const ATTENTION_ANSWER = ['Positive', 'Negative', 'Neutral'];

const INTRO =
  "Hi! I'm a high school student doing a research project on how people perceive emotion " +
  "in short online comments. You will read 11 short Reddit comments and say how the writer " +
  "seems to feel. No special knowledge is needed and there are no right or wrong answers: " +
  "I'm interested in your honest first impression. Each part of this survey uses different comments.";

function consentText() {
  return [
    'About this survey',
    '',
    'Purpose: a school research project about how people perceive emotion in short online comments.',
    '',
    'What you will do: read 11 short, real Reddit comments and choose how the writer seems to feel. It takes a few minutes.',
    '',
    'Voluntary: you can stop at any time by closing the page. Your answers are saved only if you press Submit.',
    '',
    'Anonymous: I do not ask for your name or email. Answers are used only for this project and reported in summary, never individually.',
    '',
    'Content note: the comments are real and some contain strong or rude language.',
    '',
    'Contact: ' + CONFIG.contact + ' (' + CONFIG.school + ').',
    '',
    'You must be 18 or older to take part.'
  ].join('\n');
}

const INSTRUCTIONS = [
  'For each comment, choose how the WRITER seems to feel (not how you feel about it).',
  'Go with your first impression, answer on your own, and please do not look anything up.',
  '',
  'Positive: e.g. joy, gratitude, admiration, amusement',
  'Negative: e.g. anger, sadness, disgust, disappointment',
  'Ambiguous: feelings that can go either way depending on context, e.g. surprise, curiosity, confusion, realization',
  'Neutral: no clear emotion; factual or informational',
  "I can't tell: the comment is too unclear to judge",
  '',
  'The second question under each comment is optional: tick up to 3 emotions that you see.'
].join('\n');

const CONFIRMATION =
  'Thank you very much for helping with my project! Your answers have been recorded. ' +
  'If you would like to help more, other parts of this survey (with different comments) ' +
  'are available wherever you found this link.';

function buildForms() {
  const files = DriveApp.getFilesByName(CONFIG.csvName);
  if (!files.hasNext()) throw new Error('Could not find ' + CONFIG.csvName + ' in Drive.');
  const rows = Utilities.parseCsv(files.next().getBlob().getDataAsString());
  const header = rows.shift();
  const iBlock = header.indexOf('block');
  const iCode = header.indexOf('code');
  const iText = header.indexOf('text');
  if (iBlock < 0 || iCode < 0 || iText < 0) throw new Error('CSV needs the columns: block, code, text');

  const blocks = {};
  rows.forEach(function (r) {
    if (r.length < header.length) return; // skip empty trailing lines
    (blocks[r[iBlock]] = blocks[r[iBlock]] || []).push({ code: r[iCode], text: r[iText] });
  });

  const ss = SpreadsheetApp.create('Panel responses');
  const links = ss.getSheets()[0];
  links.setName('links');
  links.appendRow(['block', 'share link', 'edit link', 'attention check: correct answer']);

  Object.keys(blocks)
    .sort(function (a, b) { return Number(a) - Number(b); })
    .forEach(function (b) {
      if (CONFIG.testBlockOnly && Number(b) !== CONFIG.testBlockOnly) return;
      const form = buildOne(Number(b), blocks[b]);
      form.setDestination(FormApp.DestinationType.SPREADSHEET, ss.getId());
      const answer = ATTENTION_ANSWER[(Number(b) - 1) % ATTENTION.length];
      links.appendRow([b, form.getPublishedUrl(), form.getEditUrl(), answer]);
      Logger.log('Block ' + b + ': ' + form.getPublishedUrl());
    });

  Logger.log('Spreadsheet: ' + ss.getUrl());
}

function buildOne(b, items) {
  const form = FormApp.create('Emotion in online comments (part ' + b + ' of ' + CONFIG.totalBlocks + ')');
  form.setDescription(INTRO);
  form.setCollectEmail(false);
  form.setLimitOneResponsePerUser(false); // true would force a Google sign-in
  form.setAllowResponseEdits(false);
  form.setProgressBar(true);
  form.setConfirmationMessage(CONFIRMATION);

  // Page 1: consent (two separate required boxes, so both must be ticked)
  form.addSectionHeaderItem().setTitle('Consent').setHelpText(consentText());
  form.addCheckboxItem()
    .setTitle('Eligibility')
    .setChoiceValues(['I am 18 or older.'])
    .setRequired(true);
  form.addCheckboxItem()
    .setTitle('Consent')
    .setChoiceValues(['I have read the information above, I understand it, and I agree to take part.'])
    .setRequired(true);

  // Page 2: instructions + comments (attention check inserted after the 5th comment)
  form.addPageBreakItem().setTitle('How to answer').setHelpText(INSTRUCTIONS);
  const all = items.slice();
  const attIndex = Math.min(5, all.length);
  all.splice(attIndex, 0, {
    code: 'B' + pad2(b) + '-ATT',
    text: ATTENTION[(b - 1) % ATTENTION.length]
  });
  all.forEach(function (it, i) { addComment(form, it, i + 1, all.length); });

  // Page 3: optional background questions
  form.addPageBreakItem().setTitle('About you (all optional)');
  form.addTextItem().setTitle('What is your first language?');
  form.addMultipleChoiceItem()
    .setTitle('How would you rate your English?')
    .setChoiceValues(['1 - Basic', '2 - Intermediate', '3 - Good', '4 - Very good', '5 - Native or near-native']);
  form.addMultipleChoiceItem()
    .setTitle('Where did you find this survey?')
    .setChoiceValues(['Friends or school', 'Reddit', 'Discord', 'Other']);

  return form;
}

function addComment(form, it, n, total) {
  form.addSectionHeaderItem()
    .setTitle('Comment ' + n + ' of ' + total)
    .setHelpText(it.text);

  form.addMultipleChoiceItem()
    .setTitle('[' + it.code + '] How does the writer of this comment seem to feel?')
    .setChoiceValues(GROUP_CHOICES)
    .setRequired(true);

  const emo = form.addCheckboxItem()
    .setTitle('[' + it.code + '] Which emotions do you see? (optional, tick up to 3)')
    .setHelpText('Listed in this order: positive, negative, ambiguous, neutral.')
    .setChoiceValues(EMOTIONS);
  emo.setValidation(
    FormApp.createCheckboxValidation()
      .setHelpText('Please tick at most 3.')
      .requireSelectAtMost(3)
      .build()
  );
}

function pad2(n) {
  return (n < 10 ? '0' : '') + n;
}
