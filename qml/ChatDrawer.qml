import QtCore
import QtQuick
import QtQuick.Controls
import QtQuick.Controls.Basic
import QtQuick.Layouts
import llm
import download
import network

Drawer {
    id: chatDrawer
    modal: false
    opacity: 0.9

    Theme {
        id: theme
    }

    signal downloadClicked

    background: Rectangle {
        height: parent.height
        color: theme.backgroundDarkest
    }

    Item {
        anchors.fill: parent
        anchors.margins: 10

        Accessible.role: Accessible.Pane
        Accessible.name: qsTr("Drawer on the left of the application")
        Accessible.description: qsTr("Drawer that is revealed by pressing the hamburger button")

        Button {
            id: newChat
            anchors.left: parent.left
            anchors.right: parent.right
            padding: 15
            font.pixelSize: theme.fontSizeLarger
            background: Rectangle {
                color: theme.backgroundDarkest
                opacity: .5
                border.color: theme.backgroundLightest
                border.width: 1
                radius: 10
            }
            contentItem: Text {
                text: qsTr("New chat")
                horizontalAlignment: Text.AlignHCenter
                color: theme.textColor

                Accessible.role: Accessible.Button
                Accessible.name: text
                Accessible.description: qsTr("Use this to launch an external application that will check for updates to the installer")
            }
            onClicked: {
                LLM.chatListModel.addChat();
                Network.sendNewChat(LLM.chatListModel.count)
            }
        }

        ScrollView {
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.rightMargin: -10
            anchors.topMargin: 10
            anchors.top: newChat.bottom
            anchors.bottom: checkForUpdatesButton.top
            anchors.bottomMargin: 10
            ScrollBar.vertical.policy: ScrollBar.AlwaysOn

            ListView {
                id: conversationList
                anchors.fill: parent
                anchors.rightMargin: 10

                model: LLM.chatListModel

                delegate: Rectangle {
                    id: chatRectangle
                    width: conversationList.width
                    height: chatName.height
                    opacity: 0.9
                    property bool isCurrent: LLM.chatListModel.currentChat === LLM.chatListModel.get(index)
                    property bool trashQuestionDisplayed: false
                    color: index % 2 === 0 ? theme.backgroundLight : theme.backgroundLighter
                    border.width: isCurrent
                    border.color: chatName.readOnly ? theme.assistantColor : theme.userColor
                    TextField {
                        id: chatName
                        anchors.left: parent.left
                        anchors.right: buttons.left
                        color: theme.textColor
                        padding: 15
                        focus: false
                        readOnly: true
                        wrapMode: Text.NoWrap
                        hoverEnabled: false // Disable hover events on the TextArea
                        selectByMouse: false // Disable text selection in the TextArea
                        font.pixelSize: theme.fontSizeLarger
                        text: readOnly ? metrics.elidedText : name
                        horizontalAlignment: TextInput.AlignLeft
                        opacity: trashQuestionDisplayed ? 0.5 : 1.0
                        TextMetrics {
                            id: metrics
                            font: chatName.font
                            text: name
                            elide: Text.ElideRight
                            elideWidth: chatName.width - 25
                        }
                        background: Rectangle {
                            color: "transparent"
                        }
                        onEditingFinished: {
                            changeName();
                            Network.sendRenameChat()
                        }
                        function changeName() {
                            LLM.chatListModel.get(index).name = chatName.text
                            chatName.focus = false
                            chatName.readOnly = true
                            chatName.selectByMouse = false
                        }
                        TapHandler {
                            onTapped: {
                                if (isCurrent)
                                    return;
                                LLM.chatListModel.currentChat = LLM.chatListModel.get(index);
                            }
                        }
                        Accessible.role: Accessible.Button
                        Accessible.name: qsTr("Select the current chat")
                        Accessible.description: qsTr("Provides a button to select the current chat or edit the chat when in edit mode")
                    }
                    Row {
                        id: buttons
                        anchors.verticalCenter: chatName.verticalCenter
                        anchors.right: chatRectangle.right
                        anchors.rightMargin: 10
                        spacing: 10
                        Button {
                            id: editButton
                            width: 30
                            height: 30
                            visible: isCurrent
                            opacity: trashQuestionDisplayed ? 0.5 : 1.0
                            background: Image {
                                width: 30
                                height: 30
                                source: "qrc:/gpt4all/icons/edit.svg"
                            }
                            onClicked: {
                                chatName.focus = true
                                chatName.readOnly = false
                                chatName.selectByMouse = true
                            }
                            Accessible.role: Accessible.Button
                            Accessible.name: qsTr("Edit the chat name")
                            Accessible.description: qsTr("Provides a button to edit the chat name")
                        }
                        Button {
                            id: c
                            width: 30
                            height: 30
                            visible: isCurrent
                            background: Image {
                                width: 30
                                height: 30
                                source: "qrc:/gpt4all/icons/trash.svg"
                            }
                            onClicked: {
                                trashQuestionDisplayed = true
                                timer.start()
                            }
                            Accessible.role: Accessible.Button
                            Accessible.name: qsTr("Delete of the chat")
                            Accessible.description: qsTr("Provides a button to delete the chat")
                        }
                    }
                    Rectangle {
                        id: trashSureQuestion
                        anchors.top: buttons.bottom
                        anchors.topMargin: 10
                        anchors.right: buttons.right
                        width: childrenRect.width
                        height: childrenRect.height
                        color: chatRectangle.color
                        visible: isCurrent && trashQuestionDisplayed
                        opacity: 1.0
                        radius: 10
                        Row {
                            spacing: 10
                            Button {
                                id: checkMark
                                width: 30
                                height: 30
                                contentItem: Text {
                                    color: theme.textErrorColor
                                    text: "\u2713"
                                    font.pixelSize: theme.fontSizeLarger
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                background: Rectangle {
                                    width: 30
                                    height: 30
                                    color: "transparent"
                                }
                                onClicked: {
                                    LLM.chatListModel.removeChat(LLM.chatListModel.get(index))
                                    Network.sendRemoveChat()
                                }
                                Accessible.role: Accessible.Button
                                Accessible.name: qsTr("Confirm delete of the chat")
                                Accessible.description: qsTr("Provides a button to confirm delete of the chat")
                            }
                            Button {
                                id: cancel
                                width: 30
                                height: 30
                                contentItem: Text {
                                    color: theme.textColor
                                    text: "\u2715"
                                    font.pixelSize: theme.fontSizeLarger
                                    horizontalAlignment: Text.AlignHCenter
                                    verticalAlignment: Text.AlignVCenter
                                }
                                background: Rectangle {
                                    width: 30
                                    height: 30
                                    color: "transparent"
                                }
                                onClicked: {
                                    trashQuestionDisplayed = false
                                }
                                Accessible.role: Accessible.Button
                                Accessible.name: qsTr("Cancel the delete of the chat")
                                Accessible.description: qsTr("Provides a button to cancel delete of the chat")
                            }
                        }
                    }
                    Timer {
                        id: timer
                        interval: 3000; running: false; repeat: false
                        onTriggered: trashQuestionDisplayed = false
                    }
                }

                Accessible.role: Accessible.List
                Accessible.name: qsTr("List of chats")
                Accessible.description: qsTr("List of chats in the drawer dialog")
            }
        }

        /*Label {
            id: discordLink
            textFormat: Text.RichText
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: conversationList.bottom
            anchors.topMargin: 20
            wrapMode: Text.WordWrap
            text: qsTr("Check out our discord channel <a href=\"https://discord.gg/4M2QFmTt2k\">https://discord.gg/4M2QFmTt2k</a>")
            onLinkActivated: { Qt.openUrlExternally("https://discord.gg/4M2QFmTt2k") }
            color: theme.textColor
            linkColor: theme.linkColor

            Accessible.role: Accessible.Link
            Accessible.name: qsTr("Discord link")
        }

        Label {
            id: nomicProps
            textFormat: Text.RichText
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.top: discordLink.bottom
            anchors.topMargin: 20
            wrapMode: Text.WordWrap
            text: qsTr("Thanks to <a href=\"https://home.nomic.ai\">Nomic AI</a> and the community for contributing so much great data and energy!")
            onLinkActivated: { Qt.openUrlExternally("https://home.nomic.ai") }
            color: theme.textColor
            linkColor: theme.linkColor

            Accessible.role: Accessible.Paragraph
            Accessible.name: qsTr("Thank you blurb")
            Accessible.description: qsTr("Contains embedded link to https://home.nomic.ai")
        }*/

        Button {
            id: checkForUpdatesButton
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: downloadButton.top
            anchors.bottomMargin: 10
            padding: 15
            contentItem: Text {
                text: qsTr("Check for updates...")
                horizontalAlignment: Text.AlignHCenter
                color: theme.textColor

                Accessible.role: Accessible.Button
                Accessible.name: text
                Accessible.description: qsTr("Use this to launch an external application that will check for updates to the installer")
            }

            background: Rectangle {
                opacity: .5
                border.color: theme.backgroundLightest
                border.width: 1
                radius: 10
                color: theme.backgroundLight
            }

            onClicked: {
                if (!LLM.checkForUpdates())
                    checkForUpdatesError.open()
            }
        }

        Button {
            id: downloadButton
            anchors.left: parent.left
            anchors.right: parent.right
            anchors.bottom: parent.bottom
            padding: 15
            contentItem: Text {
                text: qsTr("Download new models...")
                horizontalAlignment: Text.AlignHCenter
                color: theme.textColor

                Accessible.role: Accessible.Button
                Accessible.name: text
                Accessible.description: qsTr("Use this to launch a dialog to download new models")
            }

            background: Rectangle {
                opacity: .5
                border.color: theme.backgroundLightest
                border.width: 1
                radius: 10
                color: theme.backgroundLight
            }

            onClicked: {
                downloadClicked()
            }
        }
    }
}