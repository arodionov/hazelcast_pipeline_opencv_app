package app;

import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.jet.datamodel.Tuple2;
import com.hazelcast.jet.pipeline.*;
import com.hazelcast.map.IMap;
import nu.pattern.OpenCV;
import org.opencv.core.*;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import java.io.Serializable;

import static com.hazelcast.jet.pipeline.Sinks.mapWithUpdating;
import static org.opencv.imgproc.Imgproc.FONT_HERSHEY_SIMPLEX;
import static org.opencv.imgproc.Imgproc.cvtColor;

public class CameraPipeline implements Serializable {

    static {
        OpenCV.loadLocally();
    }

    static CascadeClassifier faceDetector
            = new CascadeClassifier(CameraPipeline.class.getClassLoader().getResource("haarcascade_profileface.xml").getPath());

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
                .destroyFn(VideoCapture::release)
                .build();

        //faceDetectionBasic(streamSource); // basic face detection
        faceDetectionWithCounter(streamSource); // face detection with counter
    }

    private static void faceDetectionBasic(StreamSource<byte[]> streamSource) {
        Pipeline pipeline = Pipeline.create();
        pipeline.readFrom(streamSource)
                .withoutTimestamps()
                .map(CameraPipeline::grayscale)
                .map(CameraPipeline::faceDetect)
                .writeTo(Sinks.reliableTopic("topic"));

        HazelcastInstance hz = Hazelcast.bootstrappedInstance();
        hz.getJet().newJob(pipeline);
    }

    private static void faceDetectionWithCounter(StreamSource<byte[]> streamSource) {
        Pipeline pipeline = Pipeline.create();
        StreamStage<Tuple2<Mat, Boolean>> streamStage = pipeline.readFrom(streamSource)
                .withoutTimestamps()
                .map(CameraPipeline::faceDetectTuple);

        streamStage.map(Tuple2::f1)
                .map(isFaceDetected -> isFaceDetected ? 1 : 0)
                .writeTo(mapWithUpdating("my-map", k -> "counts", Integer::sum));

        streamStage.map(Tuple2::f0)
                .mapUsingIMap("my-map", k -> "counts",
                        (Mat mat, Integer count) -> {
                            Imgproc.putText(
                                    mat,                            // Matrix obj of the image
                                    count.toString(),               // Text to be added
                                    new Point(10, 50),        // point
                                    FONT_HERSHEY_SIMPLEX ,          // front face
                                    1,                              // front scale
                                    new Scalar(0, 0, 255),          // Scalar object for color
                                    4                               // Thickness
                            );

                            MatOfByte upbytes = new MatOfByte();
                            Imgcodecs.imencode(".jpg", mat, upbytes);
                            return upbytes.toArray();
                        })
                .writeTo(Sinks.reliableTopic("topic"));

        HazelcastInstance hz = Hazelcast.bootstrappedInstance();
        IMap<String, Integer> hzMap = hz.getMap("my-map");
        hzMap.put("counts", 0);

        hz.getJet().newJob(pipeline);
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

    private static Tuple2<Mat, Boolean> faceDetectTuple(byte[] bytes) {
        Mat mat0 = Imgcodecs.imdecode(new MatOfByte(bytes), Imgcodecs.IMREAD_REDUCED_GRAYSCALE_2);
        Mat mat = new Mat();
        cvtColor(mat0, mat, Imgcodecs.IMREAD_COLOR);

        MatOfRect faceDetections = new MatOfRect();
        faceDetector.detectMultiScale(mat, faceDetections);

        boolean isFaceDetected = false;
        for (Rect rect : faceDetections.toArray()) {
            Imgproc.rectangle(mat,
                    new Point(rect.x, rect.y),
                    new Point(rect.x + rect.width,
                            rect.y + rect.height),
                    new Scalar(0, 255, 0));
            isFaceDetected = true;
        }

        return Tuple2.tuple2(mat, isFaceDetected);
    }

    private static byte[] grayscale(byte[] bytes) {
        Mat mat = Imgcodecs.imdecode(new MatOfByte(bytes), Imgcodecs.IMREAD_REDUCED_GRAYSCALE_2);

        MatOfByte upbytes = new MatOfByte();
        Imgcodecs.imencode(".jpg", mat, upbytes);
        return upbytes.toArray();
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

}
