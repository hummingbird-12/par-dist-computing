import org.apache.kafka.clients.producer.KafkaProducer;
import org.apache.kafka.clients.producer.ProducerConfig;
import org.apache.kafka.clients.producer.ProducerRecord;

import java.util.Properties;

public class Producer {
    private final KafkaProducer<String, String> producer;
    private final String topic;

    public Producer(final String id, final String topic) {
        Properties config = new Properties();
        config.put(ProducerConfig.CLIENT_ID_CONFIG, id);
        config.put(ProducerConfig.BOOTSTRAP_SERVERS_CONFIG, "127.0.0.1:9092");
        config.put(ProducerConfig.KEY_SERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringSerializer");
        config.put(ProducerConfig.VALUE_SERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringSerializer");
        config.put(ProducerConfig.LINGER_MS_CONFIG, 1);

        producer = new KafkaProducer<>(config);
        this.topic = topic;
    }

    void produce(String value) {
        ProducerRecord<String, String> record = new ProducerRecord<>(topic, value);
        try {
            producer.send(record).get();
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println(e.getMessage());
        }
    }

    public void close() {
        producer.close();
    }
}
