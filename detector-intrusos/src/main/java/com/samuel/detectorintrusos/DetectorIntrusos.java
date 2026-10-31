package com.samuel.detectorintrusos;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Frame;
import org.bytedeco.javacv.OpenCVFrameGrabber;
import org.bytedeco.javacv.FrameGrabber;
import static org.bytedeco.opencv.global.opencv_videoio.CAP_DSHOW;
import org.bytedeco.opencv.opencv_objdetect.CascadeClassifier;
import org.bytedeco.opencv.opencv_core.Mat;
import org.bytedeco.opencv.opencv_core.RectVector;
import org.bytedeco.javacv.OpenCVFrameConverter;
import static org.bytedeco.opencv.global.opencv_imgproc.COLOR_BGR2GRAY;
import static org.bytedeco.opencv.global.opencv_imgproc.cvtColor;

public class DetectorIntrusos {

    public static void main(String[] args) {

    	OpenCVFrameGrabber grabber = new OpenCVFrameGrabber(0);
    	grabber.setVideoOption("videoio_backend", String.valueOf(CAP_DSHOW));        
    	
    	CanvasFrame canvas = null;

        try {
            grabber.start();

            canvas = new CanvasFrame(
                "Webcam - Detector de Intrusos",
                CanvasFrame.getDefaultGamma()
            );

            canvas.setDefaultCloseOperation(
                javax.swing.JFrame.DISPOSE_ON_CLOSE
            );
            
            CascadeClassifier detectorFrontal =
            	    new CascadeClassifier("src/main/resources/haarcascade_frontalface_default.xml");
            
            CascadeClassifier detectorPerfil =
            	    new CascadeClassifier("src/main/resources/haarcascade_profileface.xml");
            
            if (detectorFrontal.empty() || detectorPerfil.empty()) {
        	    System.out.println("No se pudo cargar el clasificador.");
        	    System.out.println(System.getProperty("user.dir"));
        	    return;
        	}
            
            Mat imagenGris = new Mat();
           	RectVector rostrosFrontales = new RectVector();
           	RectVector rostrosDePerfil = new RectVector();
           	
           	boolean intrusoDetectado = false;
           	long ultimaDeteccion = 0;
           	long tolerancia = 2000;

            while (canvas.isVisible()) {

                Frame frame = grabber.grab();
                OpenCVFrameConverter.ToMat converter = new OpenCVFrameConverter.ToMat();
                if (frame == null) {
                    System.out.println("No se pudo capturar el fotograma.");
                    break;
                }
                
                Mat imagenColor = converter.convert(frame);
                if (imagenColor != null && !imagenColor.empty()) {
                    
                    cvtColor(imagenColor, imagenGris, COLOR_BGR2GRAY);

                    detectorFrontal.detectMultiScale(imagenGris, rostrosFrontales);
                    detectorPerfil.detectMultiScale(imagenGris, rostrosDePerfil);

                    boolean hayRostro = rostrosFrontales.size() > 0
                            || rostrosDePerfil.size() > 0;
                    long tiempoActual = System.currentTimeMillis();
		
		            if (hayRostro) {
		               ultimaDeteccion = tiempoActual;
		
		               if (!intrusoDetectado) {
		                   System.out.println("ALERTA: INTRUSO");
		                   intrusoDetectado = true;
		               }
		           }
		
		           if (!hayRostro && intrusoDetectado
		                   && tiempoActual - ultimaDeteccion >= tolerancia) {
		
		               System.out.println("INTRUSO DESAPARECIDO");
		               intrusoDetectado = false;
		           }
                canvas.showImage(frame);
                }
           }
           
        } catch (FrameGrabber.Exception e) {
            System.err.println("Error de cámara: " + e.getMessage());

        } finally {

            try {
                grabber.stop();
            } catch (FrameGrabber.Exception e) {
                System.err.println("Error al detener la cámara.");
            }

            if (canvas != null) {
                canvas.dispose();
            }
        }
    }
}