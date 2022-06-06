package app;

import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.jet.pipeline.*;
import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import java.io.Serializable;

public class CameraPipeline implements Serializable {

    static {
        OpenCV.loadLocally();
    }
    static CascadeClassifier faceDetector
            = new CascadeClassifier("resources/haarcascade_profileface.xml");

    public static void main(String[] args) {

        StreamSource<byte[]> streamSource =
                SourceBuilder.stream("video", cap -> new VideoCapture(0))
                .<byte[]>fillBufferFn((cap, buf) -> {
                    Mat mat = new Mat();
                    cap.read(mat);

                    MatOfByte bytes = new MatOfByte();
                    Imgcodecs.imencode(".jpg", mat, bytes);
                    buf.add(bytes.toArray());
                })
                .destroyFn(cap -> cap.release())
                .build();

        Pipeline pipeline = Pipeline.create();
        pipeline.readFrom(streamSource)
                .withoutTimestamps()
                .map(CameraPipeline::color) // CameraPipeline::color, CameraPipeline::edges, CameraPipeline::faceDetect
                .writeTo(Sinks.reliableTopic("topic"));

        HazelcastInstance hz = Hazelcast.bootstrappedInstance();

        hz.getJet().newJob(pipeline);
    }

    private static byte[] color(byte[] bytes) {
        Mat mat = Imgcodecs.imdecode(new MatOfByte(bytes), Imgcodecs.IMREAD_COLOR);

        MatOfByte upbytes = new MatOfByte();
        Imgcodecs.imencode(".jpg", mat, upbytes);
        return upbytes.toArray();
    }

    private static byte[] edges(byte[] bytes) {
        Mat mat = Imgcodecs.imdecode(new MatOfByte(bytes), Imgcodecs.IMREAD_COLOR);

        Mat srcBlur = new Mat();
        Mat detectedEdges = new Mat();
        Imgproc.blur(mat, srcBlur, new Size(3,3));
        Imgproc.Canny(srcBlur, detectedEdges, 0.5, 0.5 * 3, 3, false);
        Mat dst = new Mat(mat.size(), CvType.CV_8UC3, Scalar.all(0));
        mat.copyTo(dst, detectedEdges);

        MatOfByte upbytes = new MatOfByte();
        Imgcodecs.imencode(".jpg", dst, upbytes);
        return upbytes.toArray();
    }

    private static byte[] faceDetect(byte[] bytes) {
        Mat mat = Imgcodecs.imdecode(new MatOfByte(bytes), Imgcodecs.IMREAD_COLOR);

        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(mat, faceDetections);

        for (Rect rect : faceDetections.toArray()) {
            Imgproc.rectangle(mat,
                    new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width,
                            rect.y + rect.height),
                    new Scalar(0, 255, 0));
        }

        MatOfByte upbytes = new MatOfByte();
        Imgcodecs.imencode(".jpg", mat, upbytes);
        return upbytes.toArray();
    }

    private static byte[] grayscale(byte[] bytes) {
        Mat mat = Imgcodecs.imdecode(new MatOfByte(bytes), Imgcodecs.IMREAD_GRAYSCALE);

        MatOfByte upbytes = new MatOfByte();
        Imgcodecs.imencode(".jpg", mat, upbytes);
        return upbytes.toArray();
    }

}
