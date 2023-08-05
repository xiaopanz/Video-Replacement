import java.awt.image.*;
import java.io.*;
import java.util.concurrent.atomic.AtomicReference;
import javax.swing.*;

/**
 * Plays the rgb video file.
 * @author Xiaopan Zhang
 */
public class PlayRgbVideo implements Runnable{

    /**
     * CONSTRUCTOR
     */
    public PlayRgbVideo(String fileName, PlaySound pSound){
        this.fileName = fileName;
        this.playSound = pSound;
    }

    public void run() {
        play();
    }

    private void play(){


        img = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);

        try {
            AtomicReference<File> file = new AtomicReference<>(new File(fileName));
            is = new FileInputStream(file.get());

            int frameLength = 3 * WIDTH * HEIGHT;
            long numFrames = file.get().length()/frameLength;

            JFrame frame = new JFrame();

            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setTitle("Video Player");
            frame.setSize(WIDTH,HEIGHT+80);
            JButton start = new JButton();
            start.setText("Start");
            start.setBounds(80, HEIGHT+2, 80, 40);
            start.setVisible(true);
            frame.add(start);
            start.addActionListener(e -> {
                System.out.println("Clicking Start...");
                playSound.resume();
            });
            JButton pause = new JButton();
            pause.setText("Pause");
            pause.setBounds(195, HEIGHT+2, 80, 40);
            pause.setVisible(true);
            frame.add(pause);
            pause.addActionListener(e -> {
                System.out.println("Clicking Pause...");
                playSound.pause();
            });

            JButton stop = new JButton();
            stop.setText("Stop");
            stop.setBounds(310, HEIGHT+2, 80, 40);
            stop.setVisible(true);
            frame.add(stop);
			stop.addActionListener(e -> {
                System.out.println("Clicking Stop...");
                playSound.stop();

                file.set(new File(fileName));
                try {
                    is = new FileInputStream(file.get());
                } catch (FileNotFoundException ex) {
                    throw new RuntimeException(ex);
                }
                cur_time = 0;
			});

            bytes = new byte[frameLength];

            VideoComponent component = new VideoComponent();

            double samplePerFrame = playSound.getSampleRate()/FPS;

            while (true){
                // If audio is running ahead of video, playing video without any delays
                // If video is running ahead of audio, stop playing video by busy waiting


//                int cur_time = 0;
//                while(cur_time < playSound.getPosition()/samplePerFrame) {
//                    readBytes();
//                    component.setImg(img);
//                    frame.add(component);
//                    frame.repaint();
//                    frame.setVisible(true);
//                    cur_time++;
//                }
//
//                while(cur_time > 5 + playSound.getPosition()/samplePerFrame) { }

                cur_time = 0;
                for(; cur_time < numFrames - 5; cur_time++) {
                    while(cur_time > 5 + playSound.getPosition()/samplePerFrame) { }

                    readBytes();
                    component.setImg(img);
                    frame.add(component);
                    frame.repaint();
                    frame.setVisible(true);
                }

                stop.doClick();

                System.out.println("Video Ended.");
            }

        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }

    /**
     * Reads in the bytes of raw RGB data for a frame.
     */
    private  void readBytes() {
        try {
            int offset = 0;
            int numRead = 0;
            while (offset < bytes.length && (numRead = is.read(bytes, offset, bytes.length - offset)) >= 0) {
                offset += numRead;
            }
            int ind = 0;
            for(int y = 0; y < HEIGHT; y++){
                for(int x = 0; x < WIDTH; x++){
                    byte r = bytes[ind];
                    byte g = bytes[ind + HEIGHT * WIDTH];
                    byte b = bytes[ind + HEIGHT * WIDTH * 2];

                    int pix = 0xff000000 | ((r & 0xff) << 16) | ((g & 0xff) << 8) | (b & 0xff);
                    img.setRGB(x,y,pix);
                    ind++;
                }
            }
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
    private int cur_time;

    private final int WIDTH = 480;
    private final int HEIGHT = 270;
    private final double FPS = 30;
    private PlaySound playSound;
    private String fileName;
    private InputStream is;
    private BufferedImage img;
    private byte[] bytes;
}
