package net.betzel.bytedeco.javacv.pca;

/*
 * Copyright (C) 2016 Maurice Betzel
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.bytedeco.javacpp.indexer.DoubleIndexer;
import org.bytedeco.javacpp.indexer.IntIndexer;
import org.bytedeco.javacpp.opencv_core.*;
import org.bytedeco.javacv.CanvasFrame;
import org.bytedeco.javacv.Java2DFrameConverter;
import org.bytedeco.javacv.OpenCVFrameConverter;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;

import static org.bytedeco.javacpp.opencv_core.*;
import static org.bytedeco.javacpp.opencv_imgproc.*;

/**
 * PCA with JavaCV
 * https://github.com/bytedeco/javacv
 * Based on "Introduction to Principal Component Analysis (PCA) ":
 * http://docs.opencv.org/3.0.0/d1/dee/tutorial_introduction_to_pca.html
 *
 * @author Maurice Betzel
 */

public class PCA {

    public static void main(String[] args) {
        try {
            new PCA().execute(args);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void execute(String[] args) throws Exception {
        // If no params provided, compute the defaut image
        BufferedImage bufferedImage = args.length >= 1 ? ImageIO.read(new File(args[0])) : ImageIO.read(this.getClass().getResourceAsStream("/images/shapes2.jpg"));
        System.out.println("Image type: " + bufferedImage.getType());
        // Convert BufferedImage to Mat
        Mat matrix = new OpenCVFrameConverter.ToMat().convert(new Java2DFrameConverter().convert(bufferedImage));
        printMat(matrix);
        Mat gray = new Mat();
        cvtColor(matrix, gray, COLOR_BGR2GRAY);
        //Normalize
        Mat mask = new Mat();
        Mat denoised = new Mat();
        GaussianBlur(gray, denoised, new Size(5, 5), 0);
        threshold(denoised, mask, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
        normalize(gray, gray, 0, 255, NORM_MINMAX, -1, mask);
        mask.release();
        denoised.release();
        // Convert image to binary
        Mat bin = new Mat();
        threshold(gray, bin, 150, 255, THRESH_BINARY);
        // Find contours
        Mat hierarchy = new Mat();
        MatVector contours = new MatVector();
        findContours(bin, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
        long contourCount = contours.size();
        System.out.println("Countour count " + contourCount);

        for (long i = 0; i < contourCount; ++i) {
            // Calculate the area of each contour
            Mat contour = contours.get(i);
            double area = contourArea(contour);
            // Ignore contours that are too small or too large
            if (area > 128 && area < 8192) {
                principalComponentAnalysis(contour, i, matrix);
            }
        }
        CanvasFrame canvas = new CanvasFrame("PCA", 1);
        canvas.setDefaultCloseOperation(javax.swing.JFrame.EXIT_ON_CLOSE);
        canvas.setCanvasSize(320, 240);
        OpenCVFrameConverter converter = new OpenCVFrameConverter.ToIplImage();
        canvas.showImage(converter.convert(matrix));
    }

    // contour is a one dimensional array
    private void principalComponentAnalysis(Mat contour, int entry, Mat matrix) {
        //Construct a buffer used by the pca analysis
        Mat data_pts = new Mat(contour.rows(), 2, CV_64FC1);
        IntIndexer contourIndexer = contour.createIndexer();
        DoubleIndexer data_idx = data_pts.createIndexer();
        for (int i = 0; i < contour.rows(); i++) {
            data_idx.put(i, 0, contourIndexer.get(i, 0));
            data_idx.put(i, 1, contourIndexer.get(i, 1));
        }
        contourIndexer.release();
        data_idx.release();
        //Perform PCA analysis
        ArrayList<Point2d> eigen_vecs = new ArrayList(2);
        ArrayList<Double> eigen_val = new ArrayList(2);
        org.bytedeco.javacpp.opencv_core.PCA pca_analysis = new org.bytedeco.javacpp.opencv_core.PCA(data_pts, new Mat(), CV_PCA_DATA_AS_ROW);
        Mat mean = pca_analysis.mean();
        Mat eigenVector = pca_analysis.eigenvectors();
        Mat eigenValues = pca_analysis.eigenvalues();
        DoubleIndexer mean_idx = mean.createIndexer();
        DoubleIndexer eigenVectorIndexer = eigenVector.createIndexer();
        DoubleIndexer eigenValuesIndexer = eigenValues.createIndexer();
        for (int i = 0; i < 2; ++i) {
            eigen_vecs.add(new Point2d(eigenVectorIndexer.get(i, 0), eigenVectorIndexer.get(i, 1)));
            eigen_val.add(eigenValuesIndexer.get(0, i));
        }
        double cntrX = mean_idx.get(0, 0);
        double cntrY = mean_idx.get(0, 1);
        double x1 = cntrX + 0.02 * (eigen_vecs.get(0).x() * eigen_val.get(0));
        double y1 = cntrY + 0.02 * (eigen_vecs.get(0).y() * eigen_val.get(0));
        double x2 = cntrX - 0.02 * (eigen_vecs.get(1).x() * eigen_val.get(1));
        double y2 = cntrY - 0.02 * (eigen_vecs.get(1).y() * eigen_val.get(1));
        // Draw the principal components, keep accuracy during calculations
        Point cntr = new Point((int) Math.rint(cntrX), (int) Math.rint(cntrY));
        circle(matrix, cntr, 5, new Scalar(255, 0, 255, 0));
        double radian1 = Math.atan2(cntrY - y1, cntrX - x1);
        double radian2 = Math.atan2(cntrY - y2, cntrX - x2);
        double hypotenuse1 = Math.sqrt((cntrY - y1) * (cntrY - y1) + (cntrX - x1) * (cntrX - x1));
        double hypotenuse2 = Math.sqrt((cntrY - y2) * (cntrY - y2) + (cntrX - x2) * (cntrX - x2));
        //Enhance the vector signal by a factor of 2
        double point1x = cntrX - 2 * hypotenuse1 * Math.cos(radian1);
        double point1y = cntrY - 2 * hypotenuse1 * Math.sin(radian1);
        double point2x = cntrX - 2 * hypotenuse2 * Math.cos(radian2);
        double point2y = cntrY - 2 * hypotenuse2 * Math.sin(radian2);
        drawAxis(matrix, radian1, cntr, point1x, point1y, Scalar.BLUE);
        drawAxis(matrix, radian2, cntr, point2x, point2y, Scalar.CYAN);
    }

    private void drawAxis(Mat matrix, double radian, Point cntr, double x, double y, Scalar colour) {
        Point q = new Point((int) x, (int) y);
        line(matrix, cntr, q, colour);
        // create the arrow hooks
        Point arrowHook1 = new Point((int) (q.x() + 9 * Math.cos(radian + CV_PI / 4)), (int) (q.y() + 9 * Math.sin(radian + CV_PI / 4)));
        line(matrix, arrowHook1, q, colour);
        Point arrowHook2 = new Point((int) (q.x() + 9 * Math.cos(radian - CV_PI / 4)), (int) (q.y() + 9 * Math.sin(radian - CV_PI / 4)));
        line(matrix, arrowHook2, q, colour);
    }


    public static void printMat(Mat mat) {
        System.out.println("Channels: " + mat.channels());
        System.out.println("Rows: " + mat.rows());
        System.out.println("Cols: " + mat.cols());
        System.out.println("Type: " + mat.type());
        System.out.println("Dims: " + mat.dims());
        System.out.println("Depth: " + mat.depth());
    }

}
