#ifndef RESPONSETEXT_H
#define RESPONSETEXT_H

#include <QObject>
#include <QQmlEngine>
#include <QQuickTextDocument>
#include <QSyntaxHighlighter>

class PythonCodeHighlighter;
class SyntaxHighlighter : public QSyntaxHighlighter {
    Q_OBJECT
public:
    SyntaxHighlighter(QObject *parent);
    ~SyntaxHighlighter();

    void highlightBlock(const QString &text) override;

private:
    bool m_codeBlockActive;
    PythonCodeHighlighter *m_pythonHighlighter;
};

class ResponseText : public QObject
{
    Q_OBJECT
    Q_PROPERTY(QQuickTextDocument* textDocument READ textDocument WRITE setTextDocument NOTIFY textDocumentChanged())
    QML_ELEMENT
public:
    explicit ResponseText(QObject *parent = nullptr);

    QQuickTextDocument* textDocument() const;
    void setTextDocument(QQuickTextDocument* textDocument);

    void insertCodeBlock(int codeBlockEnd, const QString &codeBlockText);

Q_SIGNALS:
    void textDocumentChanged();

private:
    QQuickTextDocument *m_textDocument;
    SyntaxHighlighter *m_syntaxHighlighter;
};

#endif // RESPONSETEXT_H
