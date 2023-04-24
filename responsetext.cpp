#include "responsetext.h"

#include <QRegularExpression>
#include <QSyntaxHighlighter>
#include <QTextCharFormat>
#include <QTextDocument>
#include <QTextDocumentFragment>
#include <QFontMetricsF>
#include <QTextTableCell>

class PythonCodeHighlighter {
public:
    PythonCodeHighlighter();
    void highlightBlock(const QString &text, const QTextCharFormat &format, QTextLayout *layout);

private:
    struct HighlightingRule {
        QRegularExpression pattern;
        QTextCharFormat format;
    };
    QVector<HighlightingRule> highlightingRules;

    QTextCharFormat keywordFormat;
    QTextCharFormat commentFormat;
    QTextCharFormat stringFormat;
};

PythonCodeHighlighter::PythonCodeHighlighter() {
    HighlightingRule rule;

    keywordFormat.setForeground(Qt::darkBlue);
    keywordFormat.setFontWeight(QFont::Bold);
    QStringList keywordPatterns = {
        "\\bdef\\b", "\\bclass\\b", "\\bif\\b", "\\belse\\b", "\\belif\\b",
        "\\bwhile\\b", "\\bfor\\b", "\\breturn\\b", "\\bprint\\b", "\\bimport\\b",
        "\\bfrom\\b", "\\bas\\b", "\\btry\\b", "\\bexcept\\b", "\\braise\\b",
        "\\bwith\\b", "\\bfinally\\b", "\\bcontinue\\b", "\\bbreak\\b", "\\bpass\\b"
    };

    for (const QString &pattern : keywordPatterns) {
        rule.pattern = QRegularExpression(pattern);
        rule.format = keywordFormat;
        highlightingRules.append(rule);
    }

    commentFormat.setForeground(Qt::darkGreen);
    rule.pattern = QRegularExpression("#[^\n]*");
    rule.format = commentFormat;
    highlightingRules.append(rule);

    stringFormat.setForeground(Qt::darkRed);
    rule.pattern = QRegularExpression("\".*?\"");
    rule.format = stringFormat;
    highlightingRules.append(rule);

    rule.pattern = QRegularExpression("\'.*?\'");
    rule.format = stringFormat;
    highlightingRules.append(rule);
}

void PythonCodeHighlighter::highlightBlock(const QString &text, const QTextCharFormat &format, QTextLayout *layout) {
    QVector<QTextLayout::FormatRange> formatRanges;

    for (const HighlightingRule &rule : qAsConst(highlightingRules)) {
        QRegularExpressionMatchIterator matchIterator = rule.pattern.globalMatch(text);
        while (matchIterator.hasNext()) {
            QRegularExpressionMatch match = matchIterator.next();
            int startIndex = match.capturedStart();
            int length = match.capturedLength();
            QTextCharFormat newFormat = format;
            newFormat.merge(rule.format);

            QTextLayout::FormatRange range;
            range.start = startIndex;
            range.length = length;
            range.format = newFormat;

            formatRanges.append(range);
        }
    }

    layout->setFormats(formatRanges);
}

SyntaxHighlighter::SyntaxHighlighter(QObject *parent)
    : QSyntaxHighlighter(parent)
    , m_codeBlockActive(false)
    , m_pythonHighlighter(new PythonCodeHighlighter())
{
}

SyntaxHighlighter::~SyntaxHighlighter()
{
    delete m_pythonHighlighter;
    m_pythonHighlighter = nullptr;
}

void SyntaxHighlighter::highlightBlock(const QString &text) {
    static const QRegularExpression codeBlockMarker(R"(^```)");
    static const QRegularExpression endCodeBlockMarker(R"(^```$)");

    int state = previousBlockState();
    int startIndex = 0;

    QRegularExpressionMatch match = codeBlockMarker.match(text);
    if (match.hasMatch()) {
        state = (state == -1) ? 1 : 0;
        startIndex += match.capturedEnd();

        QTextBlock block = currentBlock();
        QTextCursor cursor(block);

        // Move the cursor to the end of the code block marker
        cursor.movePosition(QTextCursor::Right, QTextCursor::MoveAnchor, match.capturedEnd());

        // If a code block is active, end it
        if (m_codeBlockActive) {
            // End the code block and reset the active flag
            m_codeBlockActive = false;
            int codeBlockEnd = block.position() + startIndex;
            int codeBlockLength = cursor.position() - codeBlockEnd - 3; // Subtract 3 to exclude the ``` at the end
            QString codeBlockText = block.text().mid(startIndex, codeBlockLength);
            ResponseText *responseText = qobject_cast<ResponseText *>(parent());
            if (responseText) {
                responseText->insertCodeBlock(codeBlockEnd, codeBlockText);
            }
        } else {
            // Start a new code block
            m_codeBlockActive = true;
        }
    }

    if (m_codeBlockActive) {
        QTextCharFormat format;
        format.setFontFamilies(QStringList() << "Monospace");
        setFormat(0, text.length(), format);
    } else {
        // Check if there's an end code block marker at the beginning of the text
        match = endCodeBlockMarker.match(text);
        if (match.hasMatch()) {
            state = (state == -1) ? 1 : 0;
        }
    }

    setCurrentBlockState(state);
}

ResponseText::ResponseText(QObject *parent)
    : QObject{parent}
    , m_textDocument(nullptr)
    , m_syntaxHighlighter(new SyntaxHighlighter(this))
{
}

QQuickTextDocument* ResponseText::textDocument() const
{
    return m_textDocument;
}

void ResponseText::setTextDocument(QQuickTextDocument* textDocument)
{
    m_textDocument = textDocument;
    m_syntaxHighlighter->setDocument(m_textDocument->textDocument());
}

void ResponseText::insertCodeBlock(int codeBlockEnd, const QString &codeBlockText)
{
    QTextCursor cursor(m_textDocument->textDocument());
    cursor.setPosition(codeBlockEnd);

    QTextTableFormat tableFormat;
    tableFormat.setCellSpacing(0);
    tableFormat.setCellPadding(0);
    tableFormat.setWidth(QTextLength(QTextLength::FixedLength,
        80 * QFontMetricsF(cursor.charFormat().font()).averageCharWidth()));
    tableFormat.setBorder(0);

    QTextTable *table = cursor.insertTable(1, 1, tableFormat);
    QTextTableCell cell = table->cellAt(0, 0);

    QTextCharFormat cellFormat = cell.format();
    cellFormat.setBackground(Qt::black);
    cell.setFormat(cellFormat);

    QTextCursor cellCursor = cell.firstCursorPosition();
    QTextCharFormat codeFormat = cellCursor.charFormat();
    codeFormat.setForeground(Qt::white);
    codeFormat.setFontFamilies(QStringList() << "Monospace");
    cellCursor.setCharFormat(codeFormat);

    cellCursor.insertText(codeBlockText);
}