package app;

import java.awt.Image;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import com.hazelcast.client.HazelcastClient;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.topic.ITopic;
import nu.pattern.OpenCV;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.imgcodecs.Imgcodecs;

public class VideoClientApp
{
    static{
        OpenCV.loadLocally();
    }

    private static JFrame frame;
    private static JLabel imageLabel;

    public static void main(String[] args) {
        VideoClientApp app = new VideoClientApp();
        app.initGUI();

        HazelcastInstance client = HazelcastClient.newHazelcastClient();
        ITopic<byte[]> topic = client.getReliableTopic("topic");

        ImageProcessor imageProcessor = new ImageProcessor();

        topic.addMessageListener(message -> {
            byte[] bytes = message.getMessageObject();
            Mat mat = Imgcodecs.imdecode(new MatOfByte(bytes), Imgcodecs.IMREAD_COLOR);
            Image tempImage = imageProcessor.toBufferedImage(mat);
            ImageIcon imageIcon = new ImageIcon(tempImage, "Video playback");
            imageLabel.setIcon(imageIcon);
            frame.pack();
        });
    }

    private void initGUI() {
        frame = new JFrame("Video Playback Example");
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setSize(400,400);
        imageLabel = new JLabel();
        frame.add(imageLabel);
        frame.setVisible(true);
    }
}