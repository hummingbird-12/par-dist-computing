import org.apache.kafka.clients.consumer.*;
import org.apache.kafka.common.TopicPartition;

import java.time.Duration;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Consumer extends Thread {
    private final KafkaConsumer<String, String> consumer;
    private final String topic;
    private final boolean resetOffset;
    private final Map<TopicPartition, OffsetAndMetadata> currentOffsets = new HashMap<>();

    volatile ArrayList<String> data;
    static volatile boolean poll;

    Consumer(final String gid, final String topic, final boolean resetOffset) {
        Properties config = new Properties();
        config.put(ConsumerConfig.GROUP_ID_CONFIG, gid);
        config.put(ConsumerConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
        config.put(ConsumerConfig.KEY_DESERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringDeserializer");
        config.put(ConsumerConfig.VALUE_DESERIALIZER_CLASS_CONFIG, "org.apache.kafka.common.serialization.StringDeserializer");

        consumer = new KafkaConsumer<>(config);
        this.topic = topic;
        this.resetOffset = resetOffset;
        data = new ArrayList<>();
        consumer.subscribe(Collections.singletonList(topic), new RebalanceHandler());
        poll = true;
    }

    public void run() {
        final Duration timeout = Duration.ofMillis(1000);
        ConsumerRecords<String, String> records;

        try {
            while (poll) {
                records = consumer.poll(timeout);
                if (records != null && !records.isEmpty()) {
                    records.forEach(r -> {
                        currentOffsets.put(
                                new TopicPartition(r.topic(), r.partition()),
                                new OffsetAndMetadata(r.offset() + 1, "no metadata"));
                        data.add(r.value());
                    });
                    consumer.commitAsync(currentOffsets, null);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
            System.out.println(e.getMessage());
        }
    }

    private void reset() {
        consumer.seekToBeginning(
                Stream.of(new TopicPartition(topic, 0)).collect(Collectors.toList())
        );
    }

    public void close() {
        consumer.close();
    }

    private class RebalanceHandler implements ConsumerRebalanceListener {
        public void onPartitionsAssigned(Collection<TopicPartition> partitions) {
            if (resetOffset) {
                reset();
            }
        }

        public void onPartitionsRevoked(Collection<TopicPartition> partitions) {
            consumer.commitSync(currentOffsets);
        }
    }
}
