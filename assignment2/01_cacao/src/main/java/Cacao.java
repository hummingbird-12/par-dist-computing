import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

public class Cacao {
    public static final String promptPrefix = "cacaotalk> ";

    public void start() {
        Login loginWindow = new Login();
        while (loginWindow.start()) {
            final String user = loginWindow.user;
            System.out.println();
            Chatting chattingWindow = new Chatting(user);
            while (chattingWindow.start()) {
                final String chat = chattingWindow.chat;
                System.out.println();
                ChatRoom chatRoomWindow = new ChatRoom(user, chat);
                chatRoomWindow.start();
            }
        }
    }

    public static String readInput(final String prompt) {
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
        String input = "";
        try {
            System.out.print(prompt);
            input = br.readLine();
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println(e.getMessage());
        }
        return input;
    }

    public static void main(String[] args) {
        Cacao cacao = new Cacao();
        cacao.start();
    }
}

class Login {
    String user;

    public boolean start() {
        user = "";
        boolean loop;

        System.out.println("Welcome to CacaoTalk");
        System.out.println("1. Log in");
        System.out.println("2. Exit");
        System.out.println();
        do {
            loop = false;
            final String option = Cacao.readInput(Cacao.promptPrefix);
            switch (option) {
                case "1": // Log in
                    user = Cacao.readInput(Cacao.promptPrefix + "ID: ");
                    break;
                case "2": // Exit
                    finish();
                    break;
                default:
                    System.out.println("Invalid option");
                    System.out.println();
                    loop = true;
                    break;
            }
        } while (loop);

        return !user.equals("");
    }

    private void finish() {
        System.out.println("Bye.");
    }
}

class Client {
    private final Producer producer;
    private Consumer consumer;
    private final String topic;
    private final String consumerGID;

    Client(final String topic, final String prodID, final String consGID, final boolean reset) {
        producer = new Producer(prodID, topic);
        this.topic = topic;
        consumerGID = consGID;

        startPolling(reset);
    }

    void finish() {
        stopPolling();
        producer.close();
    }

    void printRecords(final boolean resetOffset) {
        ArrayList<String> data = getRecords(resetOffset);
        data.forEach(System.out::println);
        if (data.isEmpty()) {
            System.out.println();
        }
    }

    ArrayList<String> getRecords(final boolean resetOffset) {
        stopPolling();
        ArrayList<String> data = new ArrayList<>(consumer.data);
        startPolling(resetOffset);
        return data;
    }

    void putRecord(final String data) {
        producer.produce(data);
    }

    void reset() {
        stopPolling();
        startPolling(true);
    }

    private void startPolling(final boolean resetOffset) {
        consumer = new Consumer(consumerGID, topic, resetOffset);
        consumer.start();
    }

    private void stopPolling() {
        Consumer.poll = false;
        try {
            consumer.join();
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println(e.getMessage());
        }
        consumer.close();
    }
}

class Chatting {
    String user;
    String chat;
    Client client;

    Chatting(final String user) {
        this.user = user;
        client = new Client(user.concat("-chats"), "chats", "chats", true);
    }

    public boolean start() {
        chat = "";
        boolean loop;

        System.out.printf("%s's Chatting\n", user);
        System.out.println("1. List");
        System.out.println("2. Make");
        System.out.println("3. Join");
        System.out.println("4. Log out");
        System.out.println();
        do {
            loop = true;
            final String option = Cacao.readInput(Cacao.promptPrefix);
            switch (option) {
                case "1": // List
                    listChats();
                    break;
                case "2": // Make
                    if (!createChat(Cacao.readInput(Cacao.promptPrefix + "Chat room name: "))) {
                        System.out.println("Chat is already made");
                    }
                    break;
                case "3": // Join
                    chat = Cacao.readInput(Cacao.promptPrefix + "Chat room name: ");
                    if (!joinChat(chat)) {
                        chat = "";
                        System.out.println("Chat not found");
                        System.out.println();
                    }
                    else {
                        loop = false;
                    }
                    break;
                case "4": // Log out
                    loop = false;
                    break;
                default:
                    System.out.println("Invalid option");
                    System.out.println();
                    break;
            }
        } while (loop);

        client.finish();
        return !chat.equals("");
    }

    private void listChats() {
        client.printRecords(true);
    }

    private boolean createChat(final String name) {
        if (client.getRecords(true).contains(name)) {
            return false;
        }
        client.putRecord(name);
        System.out.printf("\"%s\" is created!\n", name);
        return true;
    }

    private boolean joinChat(final String name) {
        return client.getRecords(true).contains(name);
    }
}

class ChatRoom {
    String user;
    String chat;
    Client client;

    ChatRoom(final String user, final String chat) {
        this.user = user;
        this.chat = chat;
        client = new Client(chat, "chats", user, true);
    }

    public void start() {
        boolean loop;

        System.out.println(chat);
        System.out.println("1. Read");
        System.out.println("2. Write");
        System.out.println("3. Reset");
        System.out.println("4. Exit");
        System.out.println();
        do {
            loop = true;
            final String option = Cacao.readInput(Cacao.promptPrefix);
            switch (option) {
                case "1": // Read
                    readMessages();
                    break;
                case "2": // Write
                    String message = Cacao.readInput(Cacao.promptPrefix + "Text: ");
                    writeMessage(user.concat(": " + message));
                    break;
                case "3": // Reset
                    resetRoom();
                    break;
                case "4": // Log out
                    loop = false;
                    break;
                default:
                    System.out.println("Invalid option");
                    System.out.println();
                    break;
            }
        } while (loop);

        client.finish();
    }

    private void readMessages() {
        client.printRecords(false);
    }

    private void writeMessage(final String msg) {
        client.putRecord(msg);
    }

    private void resetRoom() {
        client.reset();
    }
}
