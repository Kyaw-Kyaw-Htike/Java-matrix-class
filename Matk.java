/**
 * Copyright (C) 2018 Kyaw Kyaw Htike @ Ali Abdul Ghafur. All rights reserved.
 *
 */

package KKH.StdLib;

import KKH.StdLib.Interfaces_LamdaFunctions.functor_double_doubleArray;
import org.apache.commons.math3.distribution.MultivariateNormalDistribution;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.ml.clustering.CentroidCluster;
import org.apache.commons.math3.ml.clustering.Clusterable;
import org.apache.commons.math3.ml.clustering.KMeansPlusPlusClusterer;
import org.apache.commons.math3.stat.StatUtils;
import org.apache.commons.math3.stat.descriptive.UnivariateStatistic;
import org.apache.commons.math3.stat.descriptive.moment.*;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.DoubleSupplier;

/*
3D matrix (Cube) which also naturally generalizes 2D matrices
underlying internal data is always stored in col-major (single dimensional)
double array.
*/

public final class Matk implements Serializable {

    private double[] data;
    private int nr;
    private int nc;
    private int nch;
    private int ndata_per_chan;
    private int ndata;
    
    private ImageType image_type = ImageType.NOT_IMAGE;      
    // if true, each image pixel value is assumed to range betweeen 0 to 255.
    // This is rarely the case. 
    private boolean is_image_range_0_255 = false; 
        
    public enum ImageType
    {
    	NOT_IMAGE,GRAY,RGB,RGBA,ARGB, BGR,ABGR,BGRA,BGR565,BGR555,HSV,XYZ,CrCb,Lab,Luv,HLS,LRGB,LBGR,YUV,YUV420sp,YUV420p,RGB_YV12,BGR_YV12,BGRA_YV12,RGBA_YV12,RGBA_IYUV,BGRA_IYUV,RGB_IYUV,BGR_IYUV,RGB_I420 ,BGR_I420,RGBA_I420,BGRA_I420,GRAY_420,GRAY_NV21,GRAY_NV12,GRAY_YV12,GRAY_IYUV,GRAY_I420
    }
    
    // return type of methods min() and max()    
    public static class Result_minmax
    {
        public double val;
        public int i, j, k;
    }

    // return type of methods kmeans_pp(...)
    public static class Result_clustering
    {
        public Matk centroids; // centroids, one column is for one centroid
        public Matk labels; // cluster labels for each data point
        public int nclusters; // final number of clusters after clustering
    }

    // return types of methods sort(...)
    public static class Result_sort
    {
        public Matk matSorted;
        public Matk indices_sort;
    }

    // return types of methods min(String...)
    public static class Result_minMax_eachDim
    {
        public Matk matVals;
        public Matk matIndices;
    }

    // return types of methods
    public static class Result_labelled_data
    {
        public Matk dataset;
        public Matk labels;
    }

    // return type of methods find(...)
    public static class Result_find
    {
        // linear indices assuming col major order where giving condition was found
        public int[] indices;
        // i component of ijk locations
        public int[] iPos;
        // j component of ijk locations
        public int[] jPos;
        // k component of ijk locations
        public int[] kPos;
        // the values in this matrix which satisfied the find condition
        public double[] vals;
        // number of found elements
        public int nFound;
    }
    
    /**
     * Set the info that this matrix is supposed to be intepreted as an image.
     * @param img_type
     * @param is_image_range_0_255
     */
    public void set_as_image(ImageType img_type, boolean is_image_range_0_255)
    {
    	this.image_type = img_type;
    	this.is_image_range_0_255 = is_image_range_0_255;
    	if(img_type == ImageType.GRAY && nchannels()!=1)
    		throw new IllegalArgumentException("ERROR: The specified input image type requires 1 channel. This matrix however does not have 1 channel");
    	if((img_type == ImageType.RGB || img_type == ImageType.BGR || img_type == ImageType.HSV || img_type == ImageType.XYZ || img_type == ImageType.Lab || img_type == ImageType.Luv || img_type == ImageType.HLS || img_type == ImageType.YUV) && nchannels()!=3)
    		throw new IllegalArgumentException("ERROR: The specified input image type requires 3 channels. This matrix however does not have 3 channels");
    }
    
    public boolean is_image()
    {
    	return image_type != ImageType.NOT_IMAGE;
    }
    
    public ImageType get_image_type()
    {
    	return image_type;
    }
    
    public boolean check_if_image_range_0_255()
    {
    	return is_image_range_0_255;
    }

    public Matk()
    {
        nr = 0;
        nc = 0;
        nch = 0;
        ndata_per_chan = 0;
        ndata = 0;
        data = new double[ndata];
    }

    /**
     * Create a matrix from existing data array stored in column major or row major.
     * The given data is copied to the internal storage matrix.
     * @param data contigous array
     * @param col_major if false, user's copy_array input is ignored and is always copied.
     * @param nrows
     * @param ncols
     * @param nchannels
     */
    public Matk(double[] data, boolean col_major, int nrows, int ncols, int nchannels)
    {
        if(data.length != nrows * ncols * nchannels)
            throw new IllegalArgumentException("ERROR: data.length != nr * nc * nch");

        nr = nrows;
        nc = ncols;
        nch = nchannels;
        ndata_per_chan = nr * nc;
        ndata = nrows * ncols * nchannels;

        this.data = new double[ndata];

        if(col_major)
        {
            System.arraycopy( data, 0, this.data, 0, ndata );
        }
        else
        {
        	int cc = 0;
            for(int k=0; k<nch; k++)               
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                    	this.data[cc++] = data[i * nch * nc + j * nch + k];

        }

    }

    /**
     * Create a matrix from existing data array stored in column major or row major.
     * The given data is copied to the internal storage matrix.
     * @param data contigous array
     * @param col_major if false, user's copy_array input is ignored and is always copied.
     * @param nrows
     * @param ncols
     * @param nchannels
     */
    public Matk(float[] data, boolean col_major, int nrows, int ncols, int nchannels)
    {
        if(data.length != nrows * ncols * nchannels)
            throw new IllegalArgumentException("ERROR: data.length != nr * nc * nch");

        nr = nrows;
        nc = ncols;
        nch = nchannels;
        ndata_per_chan = nr * nc;
        ndata = nrows * ncols * nchannels;

        this.data = new double[ndata];

        if(col_major)
        {
            for(int ii=0; ii<ndata; ii++)
                this.data[ii] = (double)data[ii];
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch; k++)
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        this.data[cc++] = (double)data[i * nch * nc + j * nch + k];
        }
    }

    /**
     * Create a matrix from existing data array stored in column major or row major.
     * The given data is copied to the internal storage matrix.
     * @param data contigous array
     * @param col_major if false, user's copy_array input is ignored and is always copied.
     * @param nrows
     * @param ncols
     * @param nchannels
     */
    public Matk(int[] data, boolean col_major, int nrows, int ncols, int nchannels)
    {
        if(data.length != nrows * ncols * nchannels)
            throw new IllegalArgumentException("ERROR: data.length != nr * nc * nch");

        nr = nrows;
        nc = ncols;
        nch = nchannels;
        ndata_per_chan = nr * nc;
        ndata = nrows * ncols * nchannels;

        this.data = new double[ndata];

        if(col_major)
        {
            for(int ii=0; ii<ndata; ii++)
                this.data[ii] = (double)data[ii];
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch; k++)
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        this.data[cc++] = (double)data[i * nch * nc + j * nch + k];
        }
    }

    /**
     * Create a matrix from existing data array stored in column major or row major.
     * The given data is copied to the internal storage matrix.
     * Will treat the byte as "unsigned char" which will be represented as 0-255 in the
     * internal double array.
     * @param data contigous array
     * @param col_major if false, user's copy_array input is ignored and is always copied.
     * @param nrows
     * @param ncols
     * @param nchannels
     */
    public Matk(byte[] data, boolean col_major, int nrows, int ncols, int nchannels)
    {
        if(data.length != nrows * ncols * nchannels)
            throw new IllegalArgumentException("ERROR: data.length != nr * nc * nch");

        nr = nrows;
        nc = ncols;
        nch = nchannels;
        ndata_per_chan = nr * nc;
        ndata = nrows * ncols * nchannels;

        this.data = new double[ndata];

        if(col_major)
        {
            for(int ii=0; ii<ndata; ii++)
                this.data[ii] = (double)(data[ii] & 0xFF);
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch; k++)
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        this.data[cc++] = (double)(data[i * nch * nc + j * nch + k] & 0xFF);
        }
    }

    // makes a column vector
    public Matk(double[] data)
    {
        nr = data.length;
        nc = 1;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        this.data = new double[ndata];
        System.arraycopy( data, 0, this.data, 0, ndata );

    }

    // makes a column vector
    public Matk(float[] data)
    {
        nr = data.length;
        nc = 1;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        this.data = new double[ndata];
        double[] temp = this.data;
        for(int ii=0; ii<ndata; ii++)
            temp[ii] = (double)data[ii];
    }

    // makes a column vector
    public Matk(int[] data)
    {
        nr = data.length;
        nc = 1;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        this.data = new double[ndata];
        double[] temp = this.data;
        for(int ii=0; ii<ndata; ii++)
            temp[ii] = (double)data[ii];
    }

    // makes a column vector
    public Matk(byte[] data)
    {
        nr = data.length;
        nc = 1;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        this.data = new double[ndata];
        double[] temp = this.data;
        for(int ii=0; ii<ndata; ii++)
            temp[ii] = (double)(data[ii] & 0xFF);
    }

    /**
     * Create a matrix of zeros of given dimensions
     * @param nrows
     * @param ncols
     * @param nchannels
     */
    public Matk(int nrows, int ncols, int nchannels)
    {
        nr = nrows;
        nc = ncols;
        nch = nchannels;
        ndata_per_chan = nr * nc;
        ndata = nrows * ncols * nchannels;

        data = new double[ndata];
    }

    public Matk(int nrows, int ncols)
    {
        this(nrows, ncols, 1);
    }

    public Matk(int nrows)
    {
        this(nrows, 1, 1);
    }

    public Matk(double[][] m)
    {
        nr = m.length;
        nc = m[0].length;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];
        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nr; i++)
                data[cc++] = m[i][j];
    }

    public Matk(double[][][] m)
    {
        nch = m.length;
        nr = m[0].length;
        nc = m[0][0].length;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                    data[cc++] = m[k][i][j];
    }

    public Matk(float[][] m)
    {
        nr = m.length;
        nc = m[0].length;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];
        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nr; i++)
                data[cc++] = (double)m[i][j];
    }

    public Matk(float[][][] m)
    {
        nch = m.length;
        nr = m[0].length;
        nc = m[0][0].length;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                    data[cc++] = (double)m[k][i][j];
    }

    public Matk(int[][] m)
    {
        nr = m.length;
        nc = m[0].length;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];
        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nr; i++)
                data[cc++] = (double)m[i][j];
    }

    public Matk(int[][][] m)
    {
        nch = m.length;
        nr = m[0].length;
        nc = m[0][0].length;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                    data[cc++] = (double)m[k][i][j];
    }

    public <T extends Number> Matk(List<T> data, boolean col_major, int nrows, int ncols, int nchannels)
    {
        if(data.size() != nrows * ncols * nchannels)
            throw new IllegalArgumentException("ERROR: data.length != nr * nc * nch");

        nr = nrows;
        nc = ncols;
        nch = nchannels;
        ndata_per_chan = nr * nc;
        ndata = nrows * ncols * nchannels;

        this.data = new double[ndata];

        if(col_major)
        {
            int cc = 0;
            for(int k=0; k<nch; k++)
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        this.data[cc++] = data.get(k * nr * nc + j * nr + i).doubleValue();
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch; k++)
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        this.data[cc++] = data.get(i * nch * nc + j * nch + k).doubleValue();
        }

    }

    // makes a column_vector
    public <T extends Number> Matk(List<T> data)
    {
        nr = data.size();
        nc = 1;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        this.data = new double[ndata];
        double[] temp = this.data;

        for(int ii=0; ii<ndata; ii++)
            temp[ii] = data.get(ii).doubleValue();
    }

    public <T extends Number> Matk(T[] data, boolean col_major, int nrows, int ncols, int nchannels)
    {
        if(data.length != nrows * ncols * nchannels)
            throw new IllegalArgumentException("ERROR: data.length != nr * nc * nch");

        nr = nrows;
        nc = ncols;
        nch = nchannels;
        ndata_per_chan = nr * nc;
        ndata = nrows * ncols * nchannels;

        this.data = new double[ndata];

        if(col_major)
        {
            int cc = 0;
            for(int k=0; k<nch; k++)
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        this.data[cc++] = data[k * nr * nc + j * nr + i].doubleValue();
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch; k++)
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nr; i++)
                        this.data[cc++] = data[i * nch * nc + j * nch + k].doubleValue();
        }
    }

    // makes a column_vector
    public <T extends Number> Matk(T[] data)
    {
        nr = data.length;
        nc = 1;
        nch = 1;
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        this.data = new double[ndata];
        double[] temp = this.data;

        for(int ii=0; ii<ndata; ii++)
            temp[ii] = data[ii].doubleValue();
    }

    // construct from Apache Common Math RealMatrix
    public Matk(RealMatrix m)
    {
        nch = 1;
        nr = m.getRowDimension();
        nc = m.getColumnDimension();
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];

        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nr; i++)
                data[cc++] = m.getEntry(i,j);
    }

    // construct from an array of Apache Common Math RealMatrix
    // assume that all the matrices have the same size
    public Matk(RealMatrix[] mArr)
    {
        nch = mArr.length;
        nr = mArr[0].getRowDimension();
        nc = mArr[0].getColumnDimension();
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];

        int cc = 0;
        for(int k=0; k<nch; k++)
        {
            RealMatrix m = mArr[k];
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                    data[cc++] = m.getEntry(i,j);
        }
    }

    // construct from a list of Apache Common Math RealMatrix
    // assume that all the matrices have the same size
    // dummy can be anything; just to distinguish from another method
    public Matk(List<RealMatrix> mL, boolean dummy)
    {
        nch = mL.size();
        nr = mL.get(0).getRowDimension();
        nc = mL.get(0).getColumnDimension();
        ndata_per_chan = nr * nc;
        ndata = nr * nc * nch;

        data = new double[ndata];

        int cc = 0;
        for(int k=0; k<nch; k++)
        {
            RealMatrix m = mL.get(k);
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                    data[cc++] = m.getEntry(i,j);
        }
    }

    public Matk(BufferedImage image)
    {
        nr = image.getHeight();
        nc = image.getWidth();
        int img_type_BufferedImage = image.getType();
        int cc;

        switch ( img_type_BufferedImage )
        {
            case BufferedImage.TYPE_BYTE_GRAY:
            case BufferedImage.TYPE_3BYTE_BGR:
            case BufferedImage.TYPE_4BYTE_ABGR:

                if(img_type_BufferedImage == BufferedImage.TYPE_BYTE_GRAY) nch = 1;
                else if(img_type_BufferedImage == BufferedImage.TYPE_3BYTE_BGR) nch = 3;
                else if(img_type_BufferedImage == BufferedImage.TYPE_4BYTE_ABGR) nch = 4;
                final byte[] bb = ((DataBufferByte)image.getRaster().getDataBuffer()).getData();
                ndata_per_chan = nr * nc;
                ndata = nr * nc * nch;
                data = new double[ndata];
                cc = 0;
                for(int k=0; k<nch; k++)
                    for(int j=0; j<nc; j++)
                        for(int i=0; i<nr; i++)
                            data[cc++] = ((double)(bb[i * nch * nc + j * nch + k] & 0xFF))/255.0;
                break;

            default:
            	throw new IllegalArgumentException("ERROR: Unknown type for Input BufferedImage object.");
        }
        
        ImageType img_type;
        
        switch ( img_type_BufferedImage )
        {
	        case BufferedImage.TYPE_BYTE_GRAY:
	        	img_type = ImageType.GRAY;
	        	break;
	        case BufferedImage.TYPE_3BYTE_BGR:
	        	img_type = ImageType.BGR;
	        	break;
	        case BufferedImage.TYPE_4BYTE_ABGR:
	        	img_type = ImageType.ABGR;
	        	break;
	        case BufferedImage.TYPE_USHORT_GRAY:
	        	img_type = ImageType.GRAY;
	        	break;
	        case BufferedImage.TYPE_INT_RGB:
	        	img_type = ImageType.RGB;
	        	break;
	        case BufferedImage.TYPE_INT_BGR:
	        	img_type = ImageType.BGR;
	        	break;
	        case BufferedImage.TYPE_INT_ARGB:	        	
	        	img_type = ImageType.ARGB;
	        	break; 
    		default:
    			throw new IllegalArgumentException("ERROR: Unknown type for Input BufferedImage object.");
        }
        
        set_as_image(img_type, false);
    }
    
    public static BufferedImage load_image_file_as_BufferedImage(String img_path)
    {
    	 BufferedImage img;
         try {
             img = ImageIO.read(new File(img_path));
         } catch (IOException e) {
             throw new IllegalArgumentException("ERROR: Could not read image file at img_path.");
         }    
        return img;
    }       
    
    /**
     * Construct the matrix from image file path.
     * @param img_path
     */
    public Matk(String img_path)
    {
    	this(load_image_file_as_BufferedImage(img_path));
    }
    
    /**
     * Load/deserialize this matrix from given file saved at file_path.
     * The file must have been saved with the save() method.  
     * @param file_path
     * @return
     */
    public static Matk load(String file_path)
    {
        Matk e = null;
        try {
            FileInputStream fileIn = new FileInputStream(file_path);
            ObjectInputStream in = new ObjectInputStream(fileIn);
            e = (Matk) in.readObject();
            in.close();
            fileIn.close();
            return e;
        }catch(IOException i) {
            i.printStackTrace();
            throw new IllegalArgumentException("file_path cannot be read");
        }catch(ClassNotFoundException c) {
            c.printStackTrace();
            throw new IllegalArgumentException("file_path cannot be read");
        }
    }
    
    public void save(String filepath)
    {
        try {
            FileOutputStream fileOut =
                    new FileOutputStream(filepath);
            ObjectOutputStream out = new ObjectOutputStream(fileOut);
            out.writeObject(this);
            out.close();
            fileOut.close();
            //System.out.println("Serialized data is saved in " + filepath);
        }catch(IOException i) {
            i.printStackTrace();
        }
    }

    /**
     * save col major data as a linear array in a text file
     * @param fpath
     */
    public void save_data_txt(String fpath)
    {
        try (BufferedWriter bw = new BufferedWriter(new FileWriter(fpath))) {

            bw.write(String.format("Matkc matrix: nrows = %d, ncols = %d, nchannels = %d", nr, nc, nch));
            bw.newLine();
            for(int ii=0; ii<ndata; ii++)
            {
            	bw.write(String.valueOf(data[ii]));
            	bw.newLine();
            }
                

        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    

    /**
     * Convert this matrix to BufferedImage. Makes a copy of underlying data
     * @return BufferedImage of type decided automatically.
     */
    public BufferedImage to_BufferedImage()
    {    	
    	if(!is_image())
    		throw new IllegalArgumentException("ERROR: this matrix is not an image. Therefore, cannot convert to BufferedImage.");
    	
    	if(!is_image_range_0_255)
    		mult_IP(255).min_IP(255).max_IP(0);
    	    	
    	int img_type_BufferedImage;
    	Matk mat_adjusted;
    	
    	switch(image_type)
    	{
    		case GRAY:
    			img_type_BufferedImage = BufferedImage.TYPE_BYTE_GRAY;
    			mat_adjusted = this;
    			break;
    		case BGR:
    			img_type_BufferedImage = BufferedImage.TYPE_3BYTE_BGR; 
    			mat_adjusted = this;
    			break;
    		case ABGR:
    			img_type_BufferedImage = BufferedImage.TYPE_4BYTE_ABGR; 
    			mat_adjusted = this;
    			break;
    		case RGB:
    			img_type_BufferedImage = BufferedImage.TYPE_3BYTE_BGR;
    			mat_adjusted = get_channels(new int[] {2,1,0});
    			break;
    		case ARGB:
    			img_type_BufferedImage = BufferedImage.TYPE_4BYTE_ABGR;  
    			mat_adjusted = get_channels(new int[] {0, 3, 2, 1});
    			break;
			default:
				throw new IllegalArgumentException("ERROR: Cannot convert this matrix to BufferedImage since this matrix holds an image type that has no equivalence in BufferedImage.");
    	}    	

    	BufferedImage img = new BufferedImage(ncols(), nrows(), img_type_BufferedImage);
        final byte[] targetPixels = ((DataBufferByte) img.getRaster().getDataBuffer()).getData();
        int cc = 0;
    	for(int i=0; i<nrows(); i++)
            for(int j=0; j<ncols(); j++)
            	for(int k=0; k<nchannels(); k++) 
                    targetPixels[cc++] = (byte)mat_adjusted.get(i,j,k);
    	
        return img;
    }


    public double[][][] to_double3DArray()
    {
        double[][][] mOut = new double[nch][nrows()][nc];

        int cc = 0;
        for(int k=0; k<nch; k++)
        {
            double[][] temp = mOut[k];
            for(int j=0; j<nc; j++)
                for(int i=0; i<nrows(); i++)
                    temp[i][j] = data[cc++];
        }

        return mOut;
    }

    public double[][] to_double2DArray()
    {
        if(nch != 1)
            throw new IllegalArgumentException("ERROR: nch != 1");
        double[][] mOut = new double[nrows()][nc];

        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nrows(); i++)
                mOut[i][j] = data[cc++];

        return mOut;
    }

    public double[] to_double1DArray()
    {
        if(!is_vector())
            throw new IllegalArgumentException("This matrix is not a row, col or channel vector.");

        int ndata_new = nrows() * nc * nch;
        double[] vec = new double[ndata_new];
        System.arraycopy( data, 0, vec, 0, ndata );
        return vec;
    }

    public float[][][] to_float3DArray()
    {
        float[][][] mOut = new float[nch][nrows()][nc];
            int cc = 0;
            for(int k=0; k<nch; k++)
            {
                float[][] temp = mOut[k];
                for(int j=0; j<nc; j++)
                    for(int i=0; i<nrows(); i++)
                        temp[i][j] = (float)data[cc++];
            }
        return mOut;
    }

    public float[][] to_float2DArray()
    {
        if(nch != 1)
            throw new IllegalArgumentException("ERROR: nch != 1");
        float[][] mOut = new float[nrows()][nc];

        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nrows(); i++)
                mOut[i][j] = (float)data[cc++];

        return mOut;
    }

    public float[] to_float1DArray()
    {
        if(!is_vector())
            throw new IllegalArgumentException("This matrix is not a row, col or channel vector.");

        int ndata_new = nrows() * nc * nch;
        float[] vec = new float[ndata_new];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (float)data[ii];
        return vec;
    }

    public int[][][] to_int3DArray()
    {
        int[][][] mOut = new int[nch][nrows()][nc];
        int cc = 0;
        for(int k=0; k<nch; k++)
        {
            int[][] temp = mOut[k];
            for(int j=0; j<nc; j++)
                for(int i=0; i<nrows(); i++)
                    temp[i][j] = (int)data[cc++];
        }
        return mOut;
    }

    public int[][] to_int2DArray()
    {
        if(nch != 1)
            throw new IllegalArgumentException("ERROR: nch != 1");
        int[][] mOut = new int[nrows()][nc];

        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nrows(); i++)
                mOut[i][j] = (int)data[cc++];

        return mOut;
    }

    public int[] to_int1DArray()
    {
        if(!is_vector())
            throw new IllegalArgumentException("This matrix is not a row, col or channel vector.");

        int ndata_new = nrows() * nc * nch;
        int[] vec = new int[ndata_new];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (int)data[ii];
        return vec;
    }

    public byte[][][] to_byte3DArray()
    {
        byte[][][] mOut = new byte[nch][nrows()][nc];
        int cc = 0;
        for(int k=0; k<nch; k++)
        {
            byte[][] temp = mOut[k];
            for(int j=0; j<nc; j++)
                for(int i=0; i<nrows(); i++)
                    temp[i][j] = (byte)data[cc++];
        }
        return mOut;
    }

    public byte[][] to_byte2DArray()
    {
        if(nch != 1)
            throw new IllegalArgumentException("ERROR: nch != 1");
        byte[][] mOut = new byte[nrows()][nc];
        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nrows(); i++)
                mOut[i][j] = (byte)data[cc++];
        return mOut;
    }

    public byte[] to_byte1DArray()
    {
        if(!is_vector())
            throw new IllegalArgumentException("This matrix is not a row, col or channel vector.");
        int ndata_new = nrows() * nc * nch;
        byte[] vec = new byte[ndata_new];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (byte)data[ii];
        return vec;
    }

    // convert to Apache Common Math RealMatrix
    public RealMatrix to_ACM_RealMatrix()
    {
        if(nch != 1)
            throw new IllegalArgumentException("ERROR: nch != 1");
        Array2DRowRealMatrix mOut = new Array2DRowRealMatrix(nrows(), nc);
        double[][] m = mOut.getDataRef();

        int cc = 0;
        for(int j=0; j<nc; j++)
            for(int i=0; i<nrows(); i++)
                m[i][j] = data[cc++];

        return mOut;
    }

    public void imshow(String winTitle, int x_winLocation, int y_winLocation )
    {
        BufferedImage img = to_BufferedImage();
        ImageIcon icon=new ImageIcon(img);
        JFrame frame=new JFrame(winTitle);
        JLabel lbl=new JLabel(icon);
        frame.add(lbl);
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.pack();
        frame.setLocation(x_winLocation, y_winLocation);
        frame.setVisible(true);
    }

    public void imshow(String winTitle)
    {
        imshow(winTitle, 0, 0);
    }

    public void imshow()
    {
        imshow("image", 0, 0);
    }

    /**
     * Create a matrix with uniformly distributed pseudorandom
     * integers between range [imin, imax].
     * similar to matlab's randi
     * @param nrows
     * @param ncols
     * @param nchannels
     * @param imin
     * @param imax
     * @return
     */
    public static Matk randi(int nrows, int ncols, int nchannels, int imin, int imax)
    {
        Matk mOut = new Matk(nrows, ncols, nchannels);
        Random rand = new Random();
        for(int ii=0; ii<mOut.ndata(); ii++)
            mOut.data[ii] = (double)(rand.nextInt(imax + 1 - imin) + imin);
        return mOut;
    }

    /**
     * Uniformly distributed random numbers between continuous range rangeMin and rangeMax
     * similar to matlab's rand
     * @param nrows
     * @param ncols
     * @param nchannels
     * @param rangeMin
     * @param rangeMax
     * @return
     */
    public static Matk rand(int nrows, int ncols, int nchannels, double rangeMin, double rangeMax)
    {
        if(Double.valueOf(rangeMax-rangeMin).isInfinite())
            throw new IllegalArgumentException("rangeMax-rangeMin is infinite");

        Matk mOut = new Matk(nrows, ncols, nchannels);
        Random rand = new Random();
        for(int ii=0; ii<mOut.ndata(); ii++)
            mOut.data[ii] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
        return mOut;
    }

    public static Matk rand(int nrows, int ncols, int nchannels)
    {
        return rand(nrows, ncols, nchannels, 0, 1);
    }

    /**
     * Normally distributed random numbers
     * similar to matlab's rand
     * @param nrows
     * @param ncols
     * @param nchannels
     * @param mean
     * @param std
     * @return
     */
    public static Matk randn(int nrows, int ncols, int nchannels, double mean, double std)
    {
        Matk mOut = new Matk(nrows, ncols, nchannels);
        Random rand = new Random();
        for(int ii=0; ii<mOut.ndata(); ii++)
            mOut.data[ii] = rand.nextGaussian() * std + mean;
        return mOut;
    }

    public static Matk fill_ladder(int nrows, int ncols, int nchannels, double start_val, double step)
    {
        Matk mOut = new Matk(nrows, ncols, nchannels);
        for(int ii=0; ii<mOut.ndata(); ii++)
        {
            mOut.data[ii] = start_val;
            start_val += step;
        }
        return mOut;
    }

    public static Matk fill_ladder(int nrows, int ncols, int nchannels, double start_val)
    {
        return fill_ladder(nrows, ncols, nchannels, start_val, 1);
    }

    public static Matk fill_ladder(int nrows, int ncols, int nchannels)
    {
        return fill_ladder(nrows, ncols, nchannels, 0, 1);
    }

    /**
     * Gives same results as Matlab's linspace
     * @param start_val
     * @param end_val
     * @param nvals
     * @param vector_type
     * @return
     */
    public static Matk linspace(double start_val, double end_val, int nvals, String vector_type)
    {
        Matk mOut;

        switch(vector_type)
        {
            case "row":
                mOut = new Matk(1, nvals, 1);
                break;
            case "col":
                mOut = new Matk(nvals, 1, 1);
                break;
            case "channel":
                mOut = new Matk(1, 1, nvals);
                break;
            default:
                throw new IllegalArgumentException("ERROR: vector_type must be: \"row\", \"col\" or \"channel\"");
        }

        double step = (end_val - start_val) / (nvals - 1);

        double[] temp = mOut.data;
        for(int ii=0; ii<nvals; ii++)
        {
            temp[ii] = start_val;
            start_val += step;
        }

        return mOut;
    }

    public static Matk linspace(double start_val, double end_val, int nvals)
    {
        return linspace(start_val, end_val, nvals, "row");
    }

    public static Matk randn(int nrows, int ncols, int nchannels)
    {
        return randn(nrows, ncols, nchannels, 0, 1);
    }

    public static Matk ones(int nrows, int ncols, int nchannels, double val)
    {
        Matk mOut = new Matk(nrows, ncols, nchannels);
        for(int ii=0; ii<mOut.ndata(); ii++)
            mOut.data[ii] = val;
        return mOut;
    }

    public static Matk ones(int nrows, int ncols, int nchannels)
    {
        return ones(nrows, ncols, nchannels, 1);
    }

    public static Matk zeros(int nrows, int ncols, int nchannels)
    {
        return new Matk(nrows, ncols, nchannels);
    }

    /**
     * Make a deep copy of the current matrix.
     * @return
     */
    public Matk copy_deep()
    {
        Matk mOut = new Matk(nr, nc, nch);
        System.arraycopy( data, 0, mOut.data, 0, ndata );
        return mOut;
    }

    public int nrows()
    {
        return nr;
    }

    public int ncols()
    {
        return nc;
    }

    public int nchannels()
    {
        return nch;
    }

    public int ndata() { return ndata; }

    public int ndata_per_chan() { return ndata_per_chan; }

    public int length_vec()
    {
        if(!is_vector())
            throw new IllegalArgumentException("ERROR: this matrix is not a vector");

        return ndata;
    }

    public double[] data() { return data; }

    /**
     * Find out whether this matrix is a vector (either row, column or channel vector)
     * @return
     */
    public boolean is_vector()
    {
        // a vector is a 3D matrix for which two of the dimensions has length of one.
        int z1 = nr == 1 ? 1:0;
        int z2 = nc == 1 ? 1:0;
        int z3 = nch == 1 ? 1:0;
        return z1+z2+z3 >= 2;
    }

    public boolean is_row_vector()
    {
        return nr == 1 && nc >= 1 && nch == 1;
    }

    public boolean is_col_vector()
    {
        return nr >= 1 && nc == 1 && nch == 1;
    }

    public boolean is_channel_vector()
    {
        return nr == 1 && nc == 1 && nch >= 1;
    }

    // get the copy of data corresponding to given range of a full matrix
    public Matk get(int r1, int r2, int c1, int c2, int ch1, int ch2)
    {
        if(r1 == -1) r1 = nr-1;
        if(r2 == -1) r2 = nr-1;
        if(c1 == -1) c1 = nc-1;
        if(c2 == -1) c2 = nc-1;
        if(ch1 == -1) ch1 = nch-1;
        if(ch2 == -1) ch2 = nch-1;

        int nr_new = r2 - r1 + 1;
        int nc_new = c2 - c1 + 1;
        int nch_new = ch2 - ch1 + 1;
        int ndata_per_chan_new = nr_new * nc_new;
        int ndata_new = ndata_per_chan_new * nch_new;

        Matk mOut = new Matk(nr_new, nc_new, nch_new);
        double[] temp_out = mOut.data;

        if(nr_new == nr && nc_new == nc)
        {
            System.arraycopy( data, ch1 * ndata_per_chan, temp_out, 0, ndata_new );
        }
        else if(nr_new == nr && nc_new != nc)
        {
            int cc = 0;
            for(int k=0; k<nch_new; k++)
            {
                System.arraycopy( data, (k + ch1) * ndata_per_chan + c1*nr, temp_out, cc, ndata_per_chan_new );
                cc += ndata_per_chan_new;
            }
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch_new; k++)
                for(int j=0; j<nc_new; j++)
                {
                    System.arraycopy( data, (k + ch1) * ndata_per_chan + (j + c1) * nr + r1, temp_out, cc, nr_new );
                    cc += nr_new;
                }
        }

        return mOut;
    }

    public Matk get2(int r1, int r2, int c1, int c2, int ch1, int ch2)
    {
        if(r1 == -1) r1 = nr-1;
        if(r2 == -1) r2 = nr-1;
        if(c1 == -1) c1 = nc-1;
        if(c2 == -1) c2 = nc-1;
        if(ch1 == -1) ch1 = nch-1;
        if(ch2 == -1) ch2 = nch-1;

        int nr_new = r2 - r1 + 1;
        int nc_new = c2 - c1 + 1;
        int nch_new = ch2 - ch1 + 1;
        int ndata_per_chan_new = nr_new * nc_new;
        int ndata_new = ndata_per_chan_new * nch_new;

        Matk mOut = new Matk(nr_new, nc_new, nch_new);
        double[] temp_out = mOut.data;

        int cc = 0;
        for(int k=ch1; k<=ch2; k++)
            for(int j=c1; j<=c2; j++)
                for(int i=r1; i<=r2; i++)
                    temp_out[cc++] = data[k * ndata_per_chan + j * nr + i];

        return mOut;
    }

    public double[] get_arrayOutput(int r1, int r2, int c1, int c2, int ch1, int ch2)
    {
        if(r1 == -1) r1 = nr-1;
        if(r2 == -1) r2 = nr-1;
        if(c1 == -1) c1 = nc-1;
        if(c2 == -1) c2 = nc-1;
        if(ch1 == -1) ch1 = nch-1;
        if(ch2 == -1) ch2 = nch-1;

        int nr_new = r2 - r1 + 1;
        int nc_new = c2 - c1 + 1;
        int nch_new = ch2 - ch1 + 1;
        int ndata_per_chan_new = nr_new * nc_new;
        int ndata_new = ndata_per_chan_new * nch_new;

        double[] temp_out = new double[nr_new * nc_new * nch_new];

        if(nr_new == nr && nc_new == nc)
        {
            System.arraycopy( data, ch1 * ndata_per_chan, temp_out, 0, ndata_new );
        }
        else if(nr_new == nr && nc_new != nc)
        {
            int cc = 0;
            for(int k=0; k<nch_new; k++)
            {
                System.arraycopy( data, (k + ch1) * ndata_per_chan + c1*nr, temp_out, cc, ndata_per_chan_new );
                cc += nr_new * nc_new;
            }
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch_new; k++)
                for(int j=0; j<nc_new; j++)
                {
                    System.arraycopy( data, (k + ch1) * ndata_per_chan + (j + c1) * nr + r1, temp_out, cc, nr_new );
                    cc += nr_new;
                }
        }

        return temp_out;
    }

    public Matk get(int r1, int r2, int c1, int c2)
    {
        return get(r1, r2, c1, c2, 0, -1);
    }

    public double[] get_arrayOutput(int r1, int r2, int c1, int c2)
    {
        return get_arrayOutput(r1, r2, c1, c2, 0, -1);
    }

    // get an element
    public double get(int i, int j, int k)
    {
        return data[k * ndata_per_chan + j * nr + i];
    }

    // assume k=0
    public double get(int i, int j)
    {
        return data[j * nr + i];
    }

    // get from a linear index
    public double get(int lin_index)
    {
        return data[lin_index];
    }

    // get first element
    public double get()
    {
        return data[0];
    }

    // get a discontinuous submatrix of the current matrix.
    public Matk get(int[] row_indices, int[] col_indices, int[] channel_indices)
    {
        int nr_new = row_indices.length;
        int nc_new = col_indices.length;
        int nch_new = channel_indices.length;
        Matk mOut = new Matk(nr_new, nc_new, nch_new);
        double[] temp_out = mOut.data;

        int cc = 0;
        for(int k=0; k<nch_new; k++)
            for(int j=0; j<nc_new; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[cc++] = data[channel_indices[k] * ndata_per_chan + col_indices[j] * nr + row_indices[i]];

        return mOut;
    }

    public double[] get_arrayOutput(int[] row_indices, int[] col_indices, int[] channel_indices)
    {
        int nr_new = row_indices.length;
        int nc_new = col_indices.length;
        int nch_new = channel_indices.length;
        double[] temp_out = new double[nr_new * nc_new * nch_new];

        int cc = 0;
        for(int k=0; k<nch_new; k++)
            for(int j=0; j<nc_new; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[cc++] = data[channel_indices[k] * ndata_per_chan + col_indices[j] * nr + row_indices[i]];

        return temp_out;
    }

    public Matk get_row(int row_index)
    {
        return get(row_index, row_index, 0, -1, 0, -1);
    }

    public double[] get_row_arrayOutput(int row_index)
    {
        return get_arrayOutput(row_index, row_index, 0, -1, 0, -1);
    }

    public Matk get_rows(int start_index, int end_index)
    {
        return get(start_index, end_index, 0, -1, 0, -1);
    }

    public double[] get_rows_arrayOutput(int start_index, int end_index)
    {
        return get_arrayOutput(start_index, end_index, 0, -1, 0, -1);
    }

    // take a discontinuous submatrix in the form of rows
    public Matk get_rows(int[] row_indices)
    {
        int nr_new = row_indices.length;
        Matk mOut = new Matk(nr_new, nc, nch);
        double[] temp_out = mOut.data;

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[cc++] = data[k * ndata_per_chan + j * nr + row_indices[i]];

        return mOut;
    }

    public double[] get_rows_arrayOutput(int[] row_indices)
    {
        int nr_new = row_indices.length;
        double[] temp_out = new double[nr_new * nc * nch];

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[cc++] = data[k * ndata_per_chan + j * nr + row_indices[i]];

        return temp_out;
    }

    // take a continuous submatrix in the form of a column
    public Matk get_col(int col_index)
    {
        return get(0, -1, col_index, col_index, 0, -1);
    }

    public double[] get_col_arrayOutput(int col_index)
    {
        return get_arrayOutput(0, -1, col_index, col_index, 0, -1);
    }

    // take a continuous submatrix in the form of cols
    public Matk get_cols(int start_index, int end_index)
    {
        return get(0, -1, start_index, end_index, 0, -1);
    }

    public double[] get_cols_arrayOutput(int start_index, int end_index)
    {
        return get_arrayOutput(0, -1, start_index, end_index, 0, -1);
    }

    // take a discontinuous submatrix in the form of cols
    public Matk get_cols(int[] col_indices)
    {
        int nc_new = col_indices.length;
        Matk mOut = new Matk(nr, nc_new, nch);
        double[] temp_out = mOut.data;

//        int cc = 0;
//        for(int k=0; k<nch; k++)
//            for(int j=0; j<nc_new; j++)
//                for(int i=0; i<nr; i++)
//                    temp_out[cc++] = data[k * ndata_per_chan + col_indices[j] * nr + i];

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc_new; j++)
            {
                System.arraycopy(data, k * ndata_per_chan + col_indices[j]*nr, temp_out, cc, nr);
                cc += nr;
            }

        return mOut;
    }

    public double[] get_cols_arrayOutput(int[] col_indices)
    {
        int nc_new = col_indices.length;
        double[] temp_out = new double[nr * nc_new * nch];

//        int cc = 0;
//        for(int k=0; k<nch; k++)
//            for(int j=0; j<nc_new; j++)
//                for(int i=0; i<nr; i++)
//                    temp_out[cc++] = data[k * ndata_per_chan + col_indices[j] * nr + i];

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc_new; j++)
            {
                System.arraycopy(data, k * ndata_per_chan + col_indices[j]*nr, temp_out, cc, nr);
                cc += nr;
            }

        return temp_out;
    }

    // take a continuous submatrix in the form of a channel
    public Matk get_channel(int channel_index)
    {
        return get(0, -1, 0, -1, channel_index, channel_index);
    }

    public double[] get_channel_arrayOutput(int channel_index)
    {
        return get_arrayOutput(0, -1, 0, -1, channel_index, channel_index);
    }

    // take a continuous submatrix in the form of channels
    public Matk get_channels(int start_index, int end_index)
    {
        return get(0, -1, 0, -1, start_index, end_index);
    }

    public double[] get_channels_arrayOutput(int start_index, int end_index)
    {
        return get_arrayOutput(0, -1, 0, -1, start_index, end_index);
    }

    // take a discontinuous submatrix in the form of cols
    public Matk get_channels(int[] channel_indices)
    {
        int nch_new = channel_indices.length;
        Matk mOut = new Matk(nr, nc, nch_new);
        double[] temp_out = mOut.data;

//        int cc = 0;
//        for(int k=0; k<nch_new; k++)
//            for(int j=0; j<nc; j++)
//                for(int i=0; i<nr; i++)
//                    temp_out[cc++] = data[channel_indices[k] * ndata_per_chan + j * nr + i];

        for(int k=0; k<nch_new; k++)
            System.arraycopy( data, channel_indices[k] * ndata_per_chan, temp_out, k * ndata_per_chan, ndata_per_chan );

        return mOut;
    }

    public double[] get_channels_arrayOutput(int[] channel_indices)
    {
        int nch_new = channel_indices.length;
        double[] temp_out = new double[nr * nc * nch_new];

//        int cc = 0;
//        for(int k=0; k<nch_new; k++)
//            for(int j=0; j<nc; j++)
//                for(int i=0; i<nr; i++)
//                    temp_out[cc++] = data[channel_indices[k] * ndata_per_chan + j * nr + i];

        for(int k=0; k<nch_new; k++)
            System.arraycopy( data, channel_indices[k] * ndata_per_chan, temp_out, k * ndata_per_chan, ndata_per_chan );

        return temp_out;
    }

    // use the entire given input matrix to set part of this matrix with the given range
    public Matk set(Matk mIn, int r1, int r2, int c1, int c2, int ch1, int ch2)
    {
        if(r1 == -1) r1 = nr-1;
        if(r2 == -1) r2 = nr-1;
        if(c1 == -1) c1 = nc-1;
        if(c2 == -1) c2 = nc-1;
        if(ch1 == -1) ch1 = nch-1;
        if(ch2 == -1) ch2 = nch-1;

        int nr_new = r2 - r1 + 1;
        int nc_new = c2 - c1 + 1;
        int nch_new = ch2 - ch1 + 1;
        int ndata_per_chan_new = nr_new * nc_new;
        int ndata_new = ndata_per_chan_new * nch_new;

        if(nr_new != mIn.nr || nc_new != mIn.nc || nch_new != mIn.nch)
            throw new IllegalArgumentException("ERROR: the input matrix and the range specified do not match.");

        double[] temp_in = mIn.data;
        double[] temp_out = data;

        if(nr_new == nr && nc_new == nc)
        {
            System.arraycopy( temp_in, 0, temp_out, ch1 * ndata_per_chan, ndata_new );
        }
        else if(nr_new == nr && nc_new != nc)
        {
            int cc = 0;
            for(int k=0; k<nch_new; k++)
            {
                System.arraycopy( temp_in, cc, temp_out, (k + ch1) * ndata_per_chan + c1*nr, ndata_per_chan_new );
                cc += nr_new * nc_new;
            }
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch_new; k++)
                for(int j=0; j<nc_new; j++)
                {
                    System.arraycopy( temp_in, cc, temp_out, (k + ch1) * ndata_per_chan + (j + c1) * nr + r1, nr_new );
                    cc += nr_new;
                }
        }

        return this;
    }

    // use the entire given input matrix (in the form of an
    // array stored in col major order to set part of this matrix with the given range
    public Matk set(double[] data_mIn, int r1, int r2, int c1, int c2, int ch1, int ch2)
    {
        if(r1 == -1) r1 = nr-1;
        if(r2 == -1) r2 = nr-1;
        if(c1 == -1) c1 = nc-1;
        if(c2 == -1) c2 = nc-1;
        if(ch1 == -1) ch1 = nch-1;
        if(ch2 == -1) ch2 = nch-1;

        int nr_new = r2 - r1 + 1;
        int nc_new = c2 - c1 + 1;
        int nch_new = ch2 - ch1 + 1;
        int ndata_per_chan_new = nr_new * nc_new;
        int ndata_new = ndata_per_chan_new * nch_new;

        if(ndata_new != data_mIn.length)
            throw new IllegalArgumentException("ERROR: the input matrix data and the range specified do not match.");

        double[] temp_in = data_mIn;
        double[] temp_out = data;

        if(nr_new == nr && nc_new == nc)
        {
            System.arraycopy( temp_in, 0, temp_out, ch1 * ndata_per_chan, ndata_new );
        }
        else if(nr_new == nr && nc_new != nc)
        {
            int cc = 0;
            for(int k=0; k<nch_new; k++)
            {
                System.arraycopy( temp_in, cc, temp_out, (k + ch1) * ndata_per_chan + c1*nr, ndata_per_chan_new );
                cc += nr_new * nc_new;
            }
        }
        else
        {
            int cc = 0;
            for(int k=0; k<nch_new; k++)
                for(int j=0; j<nc_new; j++)
                {
                    System.arraycopy( temp_in, cc, temp_out, (k + ch1) * ndata_per_chan + (j + c1) * nr + r1, nr_new );
                    cc += nr_new;
                }
        }

        return this;
    }

    // use the entire given input matrix to set part of this matrix with the given range
    public Matk set(Matk mIn, int r1, int r2, int c1, int c2)
    {
        return set(mIn, r1, r2, c1, c2, 0, -1);
    }

    public Matk set(double[] data_mIn, int r1, int r2, int c1, int c2)
    {
        return set(data_mIn, r1, r2, c1, c2, 0, -1);
    }

    // use the entire given input matrix to set part of this matrix starting with i,j,k position
    public Matk set(Matk mIn, int i, int j, int k)
    {
        return set(mIn, i, i+mIn.nr-1, j, j+mIn.nc-1, k, k+mIn.nch-1);
    }

    // use the entire given input matrix to set part of this matrix starting with i,j,0 position
    public Matk set(Matk mIn, int i, int j)
    {
        return set(mIn, i, i+mIn.nr-1, j, j+mIn.nc-1, 0, mIn.nch-1);
    }

    // use the given value to set an element of this matrix at i,j,k position
    public Matk set(double val, int i, int j, int k)
    {
        data[k * ndata_per_chan + j * nr + i] = val;
        return this;
    }

    // use the given value to set an element of this matrix at i,j,0 position
    public Matk set(double val, int i, int j)
    {
        data[j * nr + i] = val;
        return this;
    }

    // use the given value to set an element of this matrix at lin_index linear position
    public Matk set(double val, int lin_index)
    {
        data[lin_index] = val;
        return this;
    }

    // use the given value to set an element of this matrix at 0,0,0 position
    public Matk set(double val)
    {
        data[0] = val;
        return this;
    }

    // use the entire given input matrix to set part of this matrix specified by row, col and chan indices.
    public Matk set(Matk mIn, int[] row_indices, int[] col_indices, int[] channel_indices)
    {
        int nr_new = row_indices.length;
        int nc_new = col_indices.length;
        int nch_new = channel_indices.length;

        double[] temp_in = mIn.data;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch_new; k++)
            for(int j=0; j<nc_new; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[channel_indices[k] * ndata_per_chan + col_indices[j] * nr + row_indices[i]] = temp_in[cc++];

        return this;
    }

    public Matk set(double[] data_mIn, int[] row_indices, int[] col_indices, int[] channel_indices)
    {
        int nr_new = row_indices.length;
        int nc_new = col_indices.length;
        int nch_new = channel_indices.length;

        double[] temp_in = data_mIn;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch_new; k++)
            for(int j=0; j<nc_new; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[channel_indices[k] * ndata_per_chan + col_indices[j] * nr + row_indices[i]] = temp_in[cc++];

        return this;
    }

    // use the entire given input matrix (must be a row vector) to set a row of this matrix
    public Matk set_row(Matk mIn, int row_index)
    {
        return set(mIn, row_index, row_index, 0, -1, 0, -1);
    }

    public Matk set_row(double[] data_mIn, int row_index)
    {
        return set(data_mIn, row_index, row_index, 0, -1, 0, -1);
    }

    // use the entire given input matrix to a range of rows of this matrix
    public Matk set_rows(Matk mIn, int start_index, int end_index)
    {
        return set(mIn, start_index, end_index, 0, -1, 0, -1);
    }

    public Matk set_rows(double[] data_mIn, int start_index, int end_index)
    {
        return set(data_mIn, start_index, end_index, 0, -1, 0, -1);
    }

    // use the entire given input matrix to set specified rows of this matrix.
    public Matk set_rows(Matk mIn, int[] row_indices)
    {
        int nr_new = row_indices.length;

        double[] temp_in = mIn.data;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[k * ndata_per_chan + j * nr + row_indices[i]] = temp_in[cc++];

        return this;
    }

    public Matk set_rows(double[] data_mIn, int[] row_indices)
    {
        int nr_new = row_indices.length;

        double[] temp_in = data_mIn;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr_new; i++)
                    temp_out[k * ndata_per_chan + j * nr + row_indices[i]] = temp_in[cc++];

        return this;
    }

    // use the entire given input matrix (must be a col vector) to set a col of this matrix
    public Matk set_col(Matk mIn, int col_index)
    {
        return set(mIn, 0, -1, col_index, col_index, 0, -1);
    }

    public Matk set_col(double[] data_mIn, int col_index)
    {
        return set(data_mIn, 0, -1, col_index, col_index, 0, -1);
    }

    // use the entire given input matrix to a range of cols of this matrix
    public Matk set_cols(Matk mIn, int start_index, int end_index)
    {
        return set(mIn, 0, -1, start_index, end_index, 0, -1);
    }

    public Matk set_cols(double[] data_mIn, int start_index, int end_index)
    {
        return set(data_mIn, 0, -1, start_index, end_index, 0, -1);
    }

    // use the entire given input matrix to set specified cols of this matrix.
    public Matk set_cols(Matk mIn, int[] col_indices)
    {
        int nc_new = col_indices.length;

        double[] temp_in = mIn.data;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc_new; j++)
                for(int i=0; i<nr; i++)
                    temp_out[k * ndata_per_chan + col_indices[j] * nr + i] = temp_in[cc++];

        return this;
    }

    public Matk set_cols(double[] data_mIn, int[] col_indices)
    {
        int nc_new = col_indices.length;

        double[] temp_in = data_mIn;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch; k++)
            for(int j=0; j<nc_new; j++)
                for(int i=0; i<nr; i++)
                    temp_out[k * ndata_per_chan + col_indices[j] * nr + i] = temp_in[cc++];

        return this;
    }

    // use the entire given input matrix (must be a channel) to set a channel of this matrix
    public Matk set_channel(Matk mIn, int channel_index)
    {
        return set(mIn, 0, -1, 0, -1, channel_index, channel_index);
    }

    public Matk set_channel(double[] data_mIn, int channel_index)
    {
        return set(data_mIn, 0, -1, 0, -1, channel_index, channel_index);
    }

    // use the entire given input matrix to a range of channels of this matrix
    public Matk set_channels(Matk mIn, int start_index, int end_index)
    {
        return set(mIn, 0, -1, 0, -1, start_index, end_index);
    }

    public Matk set_channels(double[] data_mIn, int start_index, int end_index)
    {
        return set(data_mIn, 0, -1, 0, -1, start_index, end_index);
    }


    // use the entire given input matrix to set specified channels of this matrix.
    public Matk set_channels(Matk mIn, int[] channel_indices)
    {
        int nch_new = channel_indices.length;
        double[] temp_in = mIn.data;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch_new; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                    temp_out[channel_indices[k] * ndata_per_chan + j * nr + i] = temp_in[cc++];

        return this;
    }

    public Matk set_channels(double[] data_mIn, int[] channel_indices)
    {
        int nch_new = channel_indices.length;
        double[] temp_in = data_mIn;
        double[] temp_out = data;

        int cc = 0;
        for(int k=0; k<nch_new; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                    temp_out[channel_indices[k] * ndata_per_chan + j * nr + i] = temp_in[cc++];

        return this;
    }

    public void print(String name_matrix)
    {
        System.out.println("=========== Printing matrix ===========");
        for(int k=0; k<nch; k++)
        {
            System.out.println(name_matrix + "(:,:," + (k+1) + ")=[");
            for(int i=0; i<nr; i++)
            {
                for(int j=0; j<nc-1; j++)
                    System.out.print(get(i,j,k) + ",\t");
                System.out.println(get(i,nc-1,k) + ";");
            }
            System.out.println("];");
        }
        System.out.println("=========== Matrix printed ===========");
    }

    public void print()
    {
        print("mat");
    }

    public void print_info()
    {
        System.out.format("Matrix info: #rows = %d, #cols = %d, #channels = %d.\n", nr, nc, nch);
    }

    public void print_info(String name_matrix)
    {
        System.out.format("Matrix %s info: #rows = %d, #cols = %d, #channels = %d.\n", name_matrix, nr, nc, nch);
    }

    /**
     * Flatten the current matrix to either a row, column or channel vector matrix
     * Always results in a copy.
     * @param target_vec Can be "row", "column" or "channel".
     *                   If row, will result in a row vector.
     *                   If column, will result in a col vector, etc.
     * @return
     */
    public Matk vectorize(String target_vec)
    {
        Matk mOut;

        switch(target_vec)
        {
            case "row":
                mOut = new Matk(1, ndata, 1);
                break;
            case "column":
                mOut = new Matk(ndata, 1, 1);
                break;
            case "channel":
                mOut = new Matk(1, 1, ndata);
                break;
            default:
                throw new IllegalArgumentException("ERROR: target_vec must be either \"row\", \"column\" or \"channel\".");
        }

        System.arraycopy( data, 0, mOut.data, 0, ndata );
        return mOut;
    }

    /**
     * Similar as vectorize() but returns an array instead of Matkc
     * @return
     */
    public double[] vectorize_to_doubleArray()
    {
        double[] vec = new double[ndata];
        System.arraycopy( data, 0, vec, 0, ndata );
        return vec;
    }

    public float[] vectorize_to_floatArray()
    {
        float[] vec = new float[ndata];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (float)data[ii];
        return vec;
    }

    public int[] vectorize_to_intArray()
    {
        int[] vec = new int[ndata];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (int)data[ii];
        return vec;
    }

    public byte[] vectorize_to_byteArray()
    {
        byte[] vec = new byte[ndata];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (byte)data[ii];
        return vec;
    }

    public Double[] vectorize_to_DoubleArray()
    {
        Double[] vec = new Double[ndata];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = data[ii];
        return vec;
    }

    public Float[] vectorize_to_FloatArray()
    {
        Float[] vec = new Float[ndata];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (float)data[ii];
        return vec;
    }

    public Integer[] vectorize_to_IntegerArray()
    {
        Integer[] vec = new Integer[ndata];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (int)data[ii];
        return vec;
    }

    public Byte[] vectorize_to_ByteArray()
    {
        Byte[] vec = new Byte[ndata];
        for(int ii=0; ii<ndata; ii++)
            vec[ii] = (byte)data[ii];
        return vec;
    }

    /**
     * Transpose this matrix
     * @return tranposed matrix
     */
    public Matk t()
    {
        if(nch != 1)
            throw new IllegalArgumentException("ERROR: Cannot transpose matrix with more than 1 channel");

        Matk mOut = new Matk(nc, nr, 1);
        double[] temp_out = mOut.data;
        int cc = 0;
        for(int i=0; i<nr; i++)
            for(int j=0; j<nc; j++)
                temp_out[cc++] = data[j * nr + i];
        return mOut;
    }

    // reverse the channels in the matrix
    // this is useful for converting from RGB to BGR channels and vice-versa
    public Matk reverse_channels()
    {
        int[] indices_channels = new int[nch];
        for(int i=nch-1, j=0; i>=0; i--, j++)
            indices_channels[j] = i;
        return get_channels(indices_channels);
    }

    public Matk increment_IP()
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = data[ii] + 1;
        return this; // just for convenience
    }

    public Matk increment()
    {
        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = data[ii] + 1;
        return mOut;
    }

    public Matk decrement_IP()
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = data[ii] - 1;
        return this; // just for convenience
    }

    public Matk decrement()
    {
        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = data[ii] - 1;
        return mOut;
    }

    /**
     * Test whether this object is equal to anther object.
     * Two matrices are considered equal if they have
     * same nrows, ncols, nchannels and exactly the same data values.
     * This will also work for views and non-views (full matrices).
     * @param obj_
     * @return
     */
    @Override
    public boolean equals(Object obj_) {

        // If the object is compared with itself then return true
        if (obj_ == this) {
            return true;
        }

        /* Check if o is an instance of Complex or not
          "null instanceof [type]" also returns false */
        if (!(obj_ instanceof Matk)) {
            return false;
        }

        // typecast o to Complex so that we can compare data members
        Matk mIn = (Matk) obj_;

        if( (mIn.nr != nr ) || (mIn.nc != nc ) ||
                (mIn.nch != nch ))
            return false;

        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
        {
            if(data[ii] != temp_in[ii])
                return false;
        }
        return true;
    }

    /**
     * Check whether two matrices are approximately similar up to
     * some given tolerance
     * @param mIn
     * @param tolerance given tolerance. E.g. 0.00001. The smaller
     *                  this number is, the more strict the comparison
     *                  becomes.
     * @return
     */
    public boolean equals_approx(Matk mIn, double tolerance) {

        // If the object is compared with itself then return true
        if (mIn == this) {
            return true;
        }

        if( (mIn.nr != nr ) || (mIn.nc != nc ) ||
                (mIn.nch != nch ))
            return false;

        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
        {
            if(Math.abs(data[ii] - temp_in[ii]) > tolerance)
                return false;
        }

        return true;
    }

    /**
     * perform dot product between this matrix and given matrix.
     * Assume that the given matrices are column or row vectors.
     * @param mIn
     * @return
     */
    public double dot(Matk mIn)
    {
        if(ndata != mIn.ndata)
            throw new IllegalArgumentException("ERROR: ndata != mIn.ndata");

        double sum = 0;
        double temp[] = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            sum += (data[ii] * temp[ii]);

        return sum;
    }

    /**
     * element-wise multiplication of two matrices
     * @param mIn
     * @return
     */
    public Matk multE(Matk mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot element-wise multiply two matrices of different sizes.");

        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_in = mIn.data;
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = (data[ii] * temp_in[ii]);

        return mOut;
    }

    public Matk multE_IP(Matk mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot element-wise multiply two matrices of different sizes.");

        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            data[ii] = (data[ii] * temp_in[ii]);

        return this;
    }

    public Matk mult(double val)
    {
        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = data[ii] * val;

        return mOut;
    }

    public Matk mult_IP(double val)
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = data[ii] * val;
        return this;
    }

    /**
     * Multiply this matrix with another matrix.
     * @param mIn
     * @return
     */
    public Matk mult(Matk mIn)
    {
        if(nch != 1 || mIn.nch != 1)
            throw new IllegalArgumentException("ERROR: matrix multiplication can be performed on matrices with one channel.");

        if( nc != mIn.nr )
            throw new IllegalArgumentException("ERROR: Invalid sizes of matrices for mutiplication.");

        int nr_new = nr;
        int nc_new = mIn.nc;

        Matk mOut = new Matk(nr_new, nc_new, 1);
        double[] temp_out = mOut.data;
        int cc = 0;
        for(int j=0; j<nc_new; j++)
            for(int i=0; i<nr_new; i++)
                temp_out[cc++] = get_row(i).dot(mIn.get_col(j));

        return mOut;
    }

    public Matk divE(Matk mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot element-wise multiply two matrices of different sizes.");

        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_out = mOut.data;
        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = (data[ii] / temp_in[ii]);
        return mOut;
    }

    public Matk divE_IP(Matk mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot element-wise multiply two matrices of different sizes.");

        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            data[ii] = (data[ii] / temp_in[ii]);
        return this;
    }

    public Matk div(double val)
    {
        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = data[ii] / val;
        return mOut;
    }

    public Matk div_IP(double val)
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = data[ii] / val;
        return this;
    }

    public Matk pow(double val)
    {
        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = Math.pow(data[ii], val);
        return mOut;
    }

    public Matk pow_IP(double val)
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = Math.pow(data[ii], val);
        return this;
    }

    public Matk plus(Matk mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot add two matrices of different sizes.");

        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_out = mOut.data;
        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = (data[ii] + temp_in[ii]);
        return mOut;
    }

    public Matk plus_IP(Matk mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot add two matrices of different sizes.");

        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            data[ii] = (data[ii] + temp_in[ii]);
        return this;
    }

    public Matk plus(double val)
    {
        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = data[ii] + val;
        return mOut;
    }

    public Matk plus_IP(double val)
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = data[ii] + val;
        return this;
    }

    public Matk minus(Matk mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot subtract two matrices of different sizes.");

        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_out = mOut.data;
        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = (data[ii] - temp_in[ii]);
        return mOut;
    }

    public Matk minus_IP(Matk mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot subtract two matrices of different sizes.");

        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            data[ii] = (data[ii] - temp_in[ii]);
        return this;
    }

    public Matk minus(double val)
    {
        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = data[ii] - val;
        return mOut;
    }

    public Matk minus_IP(double val)
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = data[ii] - val;
        return this;
    }

    // Generate a matrix by replicating this matrix in a block-like fashion
    // similar to matlab's repmat
    public Matk repmat(int ncopies_row, int ncopies_col, int ncopies_ch)
    {
        int nrows_this = nr;
        int ncols_this = nc;
        int nchannels_this = nch;

        Matk matOut = new Matk(nrows_this*ncopies_row,
                ncols_this*ncopies_col, nchannels_this*ncopies_ch);

        int row1, row2, col1, col2, chan1, chan2;

        for (int k = 0; k < ncopies_ch; k++)
            for (int j = 0; j < ncopies_col; j++)
                for (int i = 0; i < ncopies_row; i++)
                {
                    row1 = i*nrows_this;
                    row2 = i*nrows_this + nrows_this - 1;
                    col1 = j*ncols_this;
                    col2 = j*ncols_this + ncols_this - 1;
                    chan1 = k*nchannels_this;
                    chan2 = k*nchannels_this + nchannels_this - 1;
                    matOut.set(this, row1, row2, col1, col2, chan1, chan2);
                }

        return matOut;
    }

    /**
     * Reshape a matrix.
     * @param nrows_new
     * @param ncols_new
     * @param nchannels_new
     */
    public Matk reshape(int nrows_new, int ncols_new, int nchannels_new)
    {
        if (nrows_new * ncols_new * nchannels_new != ndata)
            throw new IllegalArgumentException("ERROR: nrows_new * ncols_new * nchannels_new != ndata.");

        Matk mOut = new Matk(nrows_new, ncols_new, nchannels_new);
        System.arraycopy( data, 0, mOut.data, 0, ndata );
        return mOut;
    }

    public Matk round()
    {
        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = Math.round(data[ii]);
        return mOut;
    }

    public Matk round_IP()
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = Math.round(data[ii]);
        return this;
    }

    public Matk zeros_IP()
    {
        Arrays.fill(data, 0);
        return this;
    }

    public Matk ones_IP()
    {
        Arrays.fill(data, 1);
        return this;
    }

    public Matk fill_IP(double val)
    {
        Arrays.fill(data, val);
        return this;
    }

    /**
     * Fill this matrix with uniformly distributed pseudorandom
     * integers between range [imin, imax]
     * similar to matlab's randi
     * @param imin
     * @param imax
     * @return
     */
    public Matk randi_IP(int imin, int imax)
    {
        Random rand = new Random();
        for(int ii=0; ii<ndata; ii++)
            data[ii] = (double)(rand.nextInt(imax + 1 - imin) + imin);
        return this;
    }

    public Matk randi(int imin, int imax)
    {
        Matk mOut = new Matk(nr, nc, nch);
        Random rand = new Random();
        double[] temp_out = mOut.data;
        for(int ii=0; ii<mOut.ndata(); ii++)
            temp_out[ii] = (double)(rand.nextInt(imax + 1 - imin) + imin);
        return mOut;
    }

    /**
     * Uniformly distributed random numbers between continuous range rangeMin and rangeMax
     * similar to matlab's rand
     * @param rangeMin
     * @param rangeMax
     * @return
     */
    public Matk rand(double rangeMin, double rangeMax)
    {
        if(Double.valueOf(rangeMax-rangeMin).isInfinite())
            throw new IllegalArgumentException("rangeMax-rangeMin is infinite");

        Matk mOut = new Matk(nr, nc, nch);
        Random rand = new Random();
        double[] temp_out = mOut.data;
        for(int ii=0; ii<mOut.ndata(); ii++)
            temp_out[ii] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
        return mOut;
    }

    public Matk rand()
    {
        return rand(0, 1);
    }

    public Matk rand_IP(double rangeMin, double rangeMax)
    {
        if(Double.valueOf(rangeMax-rangeMin).isInfinite())
            throw new IllegalArgumentException("rangeMax-rangeMin is infinite");

        Random rand = new Random();
        for(int ii=0; ii<ndata; ii++)
            data[ii] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
        return this;
    }

    public Matk rand_IP()
    {
        return rand_IP(0, 1);
    }

    /**
     * Normally distributed random numbers
     * similar to matlab's randn
     * @param mean
     * @param std
     * @return
     */
    public Matk randn(double mean, double std)
    {
        Matk mOut = new Matk(nr, nc, nch);
        Random rand = new Random();
        double[] temp = mOut.data;
        for(int ii=0; ii<mOut.ndata(); ii++)
            temp[ii] = rand.nextGaussian() * std + mean;
        return mOut;
    }

    public Matk randn()
    {
        return randn(0, 1);
    }

    public Matk randn_IP(double mean, double std)
    {
        Random rand = new Random();
        for(int ii=0; ii<ndata; ii++)
            data[ii] = rand.nextGaussian() * std + mean;
        return this;
    }

    public Matk randn_IP()
    {
        return randn_IP(0, 1);
    }

    public Matk rand_custom(DoubleSupplier functor)
    {
        Matk mOut = new Matk(nr, nc, nch);
        Random rand = new Random();
        double[] temp = mOut.data;
        for(int ii=0; ii<mOut.ndata(); ii++)
            temp[ii] = functor.getAsDouble();
        return mOut;
    }

    public Matk rand_custom_IP(DoubleSupplier functor)
    {
        Random rand = new Random();
        for(int ii=0; ii<ndata; ii++)
            data[ii] = functor.getAsDouble();
        return this;
    }

    public Matk fill_ladder(double start_val, double step)
    {
        Matk mOut = new Matk(nr, nc, nch);
        double[] temp = mOut.data;
        for(int ii=0; ii<mOut.ndata(); ii++)
        {
            temp[ii] = start_val;
            start_val += step;
        }
        return mOut;
    }

    public Matk fill_ladder_IP(double start_val, double step)
    {
        for(int ii=0; ii<ndata; ii++)
        {
            data[ii] = start_val;
            start_val += step;
        }
        return this;
    }

    public Result_sort sort(boolean sort_col, boolean sort_ascend)
    {
        if(nch!=1)
            throw new IllegalArgumentException("ERROR: for sorting, this matrix must have only one channel.");

        int number_rows = nr;
        int number_cols = nc;

        Result_sort res = new Result_sort();

        res.matSorted = new Matk(number_rows, number_cols, 1);
        res.indices_sort = new Matk(number_rows, number_cols, 1);

        if (sort_col)
        {
            Double[] vals = new Double[number_rows];
            for (int j = 0; j < number_cols; j++)
            {
                Integer[] indices = stdfuncs.fill_ladder_Integer(number_rows, 0, 1);
                for(int ii=0; ii < number_rows; ii++)
                    vals[ii] = get(ii,j,0);

                if(sort_ascend)
                {
                    Arrays.sort(indices, (ee,ff)->
                            {
                                if((double)vals[(int)ee] > (double)vals[(int)ff]) return 1;
                                else if((double)vals[(int)ee] == (double)vals[(int)ff]) return 0;
                                else return -1;
                            }
                    );
                }
                else // descend
                {
                    Arrays.sort(indices, (ee,ff)->
                            {
                                if((double)vals[(int)ee] < (double)vals[(int)ff]) return 1;
                                else if((double)vals[(int)ee] == (double)vals[(int)ff]) return 0;
                                else return -1;
                            }
                    );
                }

                for(int ii=0; ii < number_rows; ii++)
                {
                    res.indices_sort.set(indices[ii], ii,j,0);
                    res.matSorted.set(vals[indices[ii]], ii,j,0);
                }
            }
        }
        else
        {
            Double[] vals = new Double[number_cols];
            for (int i = 0; i < number_rows; i++)
            {
                Integer[] indices = stdfuncs.fill_ladder_Integer(number_cols, 0, 1);
                for(int jj=0; jj < number_cols; jj++)
                    vals[jj] = get(i,jj,0);

                if(sort_ascend)
                {
                    Arrays.sort(indices, (ee,ff)->
                            {
                                if((double)vals[(int)ee] > (double)vals[(int)ff]) return 1;
                                else if((double)vals[(int)ee] == (double)vals[(int)ff]) return 0;
                                else return -1;
                            }
                    );
                }
                else // descend
                {
                    Arrays.sort(indices, (ee,ff)->
                            {
                                if((double)vals[(int)ee] < (double)vals[(int)ff]) return 1;
                                else if((double)vals[(int)ee] == (double)vals[(int)ff]) return 0;
                                else return -1;
                            }
                    );
                }

                for(int jj=0; jj < number_cols; jj++)
                {
                    res.indices_sort.set(indices[jj], i,jj,0);
                    res.matSorted.set(vals[indices[jj]], i,jj,0);
                }
            }
        }

        return res;
    }

    public Result_sort sort(boolean sort_col)
    {
        return sort(sort_col, true);
    }

    public Matk max(Matk mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: This matrix and input matrix do not have same size.");

        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_out = mOut.data;
        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = Math.max(data[ii], temp_in[ii]);
        return mOut;
    }

    public Matk max_IP(Matk mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot element-wise divide two matrices of different sizes.");

        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            data[ii] = Math.max(data[ii], temp_in[ii]);
        return this;
    }

    public Matk max(double val)
    {
        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = Math.max(data[ii], val);
        return mOut;
    }

    public Matk max_IP(double val)
    {
        for(int ii=0; ii<ndata; ii++)
            data[ii] = Math.max(data[ii], val);
        return this;
    }

    public Result_minMax_eachDim max(String process_dim)
    {
        int number_rows = nr;
        int number_cols = nc;
        int number_chans = nch;

        switch(process_dim)
        {
            case "col":
            {
                Result_minMax_eachDim res = new Result_minMax_eachDim();
                res.matVals = new Matk(1, number_cols, number_chans);
                res.matIndices = new Matk(1, number_cols, number_chans);
                double maxCur, idxMaxCur, val;
                for(int k=0; k< number_chans; k++)
                    for (int j = 0; j < number_cols; j++)
                    {
                        maxCur = get(0, j, k);
                            idxMaxCur = 0;
                            for (int i = 1; i < number_rows; i++)
                            {
                            val = get(i, j, k);
                            if (val > maxCur)
                            {
                                idxMaxCur = i;
                                maxCur = val;
                            }
                        }
                        res.matVals.set(maxCur,0, j, k);
                        res.matIndices.set(idxMaxCur,0, j, k);
                    }
                return res;
            }
            case "row":
            {
                Result_minMax_eachDim res = new Result_minMax_eachDim();
                res.matVals = new Matk(number_rows, 1, number_chans);
                res.matIndices = new Matk(number_rows, 1, number_chans);
                double maxCur, idxMaxCur, val;
                for(int k=0; k< number_chans; k++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        maxCur = get(i, 0, k);
                        idxMaxCur = 0;
                        for (int j = 1; j < number_cols; j++)
                        {
                            val = get(i, j, k);
                            if (val > maxCur)
                            {
                                idxMaxCur = j;
                                maxCur = val;
                            }
                        }
                        res.matVals.set(maxCur, i, 0, k);
                        res.matIndices.set(idxMaxCur, i, 0, k);
                    }
                return res;
            }
            case "channel":
            {
                Result_minMax_eachDim res = new Result_minMax_eachDim();
                res.matVals = new Matk(number_rows, number_cols, 1);
                res.matIndices = new Matk(number_rows, number_cols, 1);
                double maxCur, idxMaxCur, val;
                for (int j = 0; j < number_cols; j++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        maxCur = get(i, j, 0);
                        idxMaxCur = 0;
                        for(int k=1; k< number_chans; k++)
                        {
                            val = get(i, j, k);
                            if (val > maxCur)
                            {
                                idxMaxCur = k;
                                maxCur = val;
                            }
                        }
                        res.matVals.set(maxCur, i, j, 0);
                        res.matIndices.set(idxMaxCur, i, j, 0);
                    }
                return res;
            }
            default:
                throw new IllegalArgumentException("ERROR: process_dim must be \"row\", \"col\" or \"channel\".");
        }
    }

    public Result_minmax max()
    {
        Result_minmax res = new Result_minmax();
        res.val = data[0];
        res.i = 0;
        res.j = 0;
        res.k = 0;
        double val_cur;

        int cc = 0;

        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                {
                    val_cur = data[cc++];
                    if(val_cur > res.val)
                    {
                        res.val = val_cur;
                        res.i = i;
                        res.j = j;
                        res.k = k;
                    }
                }

        return res;
    }

    public Matk min(Matk mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: This matrix and input matrix do not have same size.");

        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_out = mOut.data;
        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = Math.min(data[ii], temp_in[ii]);
        return mOut;
    }

    public Matk min_IP(Matk mIn)
    {
        if( (nr!=mIn.nr) || (nc!=mIn.nc)
                || (nch!=mIn.nch))
            throw new IllegalArgumentException("ERROR: Cannot element-wise divide two matrices of different sizes.");
        double[] temp_in = mIn.data;
        for(int ii=0; ii<ndata; ii++)
            data[ii] = Math.min(data[ii], temp_in[ii]);
        return this;
    }

    public Matk min(double val)
    {
        Matk mOut = new Matk(nr, nc, nch);
        double[] temp_out = mOut.data;
        for(int ii=0; ii<ndata; ii++)
            temp_out[ii] = Math.min(data[ii], val);
        return mOut;
    }

    public Matk min_IP(double val)
    {
            for(int ii=0; ii<ndata; ii++)
                data[ii] = Math.min(data[ii], val);
        return this;
    }

    public Result_minMax_eachDim min(String process_dim)
    {
        int number_rows = nr;
        int number_cols = nc;
        int number_chans = nch;

        switch(process_dim)
        {
            case "col":
            {
                Result_minMax_eachDim res = new Result_minMax_eachDim();
                res.matVals = new Matk(1, number_cols, number_chans);
                res.matIndices = new Matk(1, number_cols, number_chans);
                double minCur, idxMinCur, val;
                for(int k=0; k< number_chans; k++)
                    for (int j = 0; j < number_cols; j++)
                    {
                        minCur = get(0, j, k);
                        idxMinCur = 0;
                        for (int i = 1; i < number_rows; i++)
                        {
                            val = get(i, j, k);
                            if (val < minCur)
                            {
                                idxMinCur = i;
                                minCur = val;
                            }
                        }
                        res.matVals.set(minCur,0, j, k);
                        res.matIndices.set(idxMinCur,0, j, k);
                    }
                return res;
            }
            case "row":
            {
                Result_minMax_eachDim res = new Result_minMax_eachDim();
                res.matVals = new Matk(number_rows, 1, number_chans);
                res.matIndices = new Matk(number_rows, 1, number_chans);
                double minCur, idxMinCur, val;
                for(int k=0; k< number_chans; k++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        minCur = get(i, 0, k);
                        idxMinCur = 0;
                        for (int j = 1; j < number_cols; j++)
                        {
                            val = get(i, j, k);
                            if (val < minCur)
                            {
                                idxMinCur = j;
                                minCur = val;
                            }
                        }
                        res.matVals.set(minCur, i, 0, k);
                        res.matIndices.set(idxMinCur, i, 0, k);
                    }
                return res;
            }
            case "channel":
            {
                Result_minMax_eachDim res = new Result_minMax_eachDim();
                res.matVals = new Matk(number_rows, number_cols, 1);
                res.matIndices = new Matk(number_rows, number_cols, 1);
                double minCur, idxMinCur, val;
                for (int j = 0; j < number_cols; j++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        minCur = get(i, j, 0);
                        idxMinCur = 0;
                        for(int k=1; k< number_chans; k++)
                        {
                            val = get(i, j, k);
                            if (val < minCur)
                            {
                                idxMinCur = k;
                                minCur = val;
                            }
                        }
                        res.matVals.set(minCur, i, j, 0);
                        res.matIndices.set(idxMinCur, i, j, 0);
                    }
                return res;
            }
            default:
                throw new IllegalArgumentException("ERROR: process_dim must be \"row\", \"col\" or \"channel\".");
        }
    }

    // compute the minimum value in the entire matrix and the corresponding index (location)
    public Result_minmax min()
    {
        Result_minmax res = new Result_minmax();
        res.val = data[0];
        res.i = 0;
        res.j = 0;
        res.k = 0;
        double val_cur;

        int cc = 0;

        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                {
                    val_cur = data[cc++];
                    if(val_cur < res.val)
                    {
                        res.val = val_cur;
                        res.i = i;
                        res.j = j;
                        res.k = k;
                    }
                }

        return res;
    }
    
    /**
     * Get the min value of all the elements in the matrix
     * @return
     */
    public double get_min_val()
    {
    	double min_val = data[0];
    	for(double cur : data)
    	{
    		if(cur < min_val)
    			min_val = cur;
    	}
    	return min_val;  
    }
    
    /**
     * Get the max value of all the elements in the matrix
     * @return
     */
    public double get_max_val()
    {
    	double max_val = data[0];
    	for(double cur: data)
    	{
    		if(cur > max_val)
    			max_val = cur;
    	}
    	return max_val;    	
    }

    /**
     * Compute univariate moments such as mean, variance, std
     * @param moment_type a string identifying the type of moment that is to be
     *                    computed. can be "mean", "std", "var", "GeometricMean",
     *                    "Kurtosis", "SecondMoment", "SemiVariance", "Skewness"
     * @param process_dim can be "col", "row" or "channel". If "col", then the
     *                    statistics for each column is computed and the results
     *                    is saved. If "row", then statics for each row is computed, etc.
     * @param isBiasCorrected only applies for "std", "var" and "SemiVariance". The
     *                        default is true. If true, compute sample statistics. If false
     *                        then population statistics. E.g. for variance, if sample
     *                        statistics, then variance = sum((x_i - mean)^2) / (n - 1) is
     *                        used to compute the statistics whereas, if population statistics,
     *                        then variance = sum((x_i - mean)^2) / n is used.
     * @return
     */
    public Matk moment(String moment_type, String process_dim, boolean isBiasCorrected)
    {
        int number_rows = nr;
        int number_cols = nc;
        int number_chans = nch;

        UnivariateStatistic statsObj;

        switch(moment_type)
        {
            case "mean":
                statsObj = new Mean();
                break;
            case "std":
                statsObj = new StandardDeviation(isBiasCorrected);
                break;
            case "var":
                statsObj = new Variance(isBiasCorrected);
                break;
            case "GeometricMean":
                statsObj = new GeometricMean();
                break;
            case "Kurtosis":
                statsObj = new Kurtosis();
                break;
            case "SecondMoment":
                statsObj = new SecondMoment();
                break;
            case "SemiVariance":
                statsObj = new SemiVariance(isBiasCorrected);
                break;
            case "Skewness":
                statsObj = new Skewness();
                break;

            default:
                throw new IllegalArgumentException("ERROR: Invalid moment_type.");
        }

        switch(process_dim)
        {
            case "col":
            {
                Matk matOut = new Matk(1, number_cols, number_chans);
                double[] vals = new double[number_rows];
                for(int k=0; k< number_chans; k++)
                    for (int j = 0; j < number_cols; j++)
                    {
                        for (int i = 0; i < number_rows; i++)
                            vals[i] = get(i, j, k);
                        matOut.set(statsObj.evaluate(vals),0,j,k);
                    }
                return matOut;
            }
            case "row":
            {
                Matk matOut = new Matk(number_rows, 1, number_chans);
                double[] vals = new double[number_cols];
                for(int k=0; k< number_chans; k++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        for (int j = 0; j < number_cols; j++)
                            vals[j] = get(i, j, k);
                        matOut.set(statsObj.evaluate(vals), i,0,k);
                    }
                return matOut;
            }
            case "channel":
            {
                Matk matOut = new Matk(number_rows, number_cols, 1);
                double[] vals = new double[number_chans];
                for (int j = 0; j < number_cols; j++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        for(int k=0; k< number_chans; k++)
                            vals[k] = get(i, j, k);
                        matOut.set(statsObj.evaluate(vals), i,j,0);
                    }
                return matOut;
            }
            default:
                throw new IllegalArgumentException("ERROR: process_dim must be \"row\", \"col\" or \"channel\".");
        }
    }

    public Matk moment(String moment_type, String process_dim)
    {
        return moment(moment_type, process_dim, true);
    }

    /**
     * Summarize each row, column or channel with a single number
     * @param process_dim can be "col", "row" or "channel". If "col", then the
     *                    summary for each column is computed and the results
     *                    is saved. If "row", then statics for each row is computed, etc.
     * @param functor a class that implements the functor_double_doubleArray interface
     *                which has only one member function "double apply(double[] x)
     * @return
     */
    public Matk summarize(String process_dim, functor_double_doubleArray functor)
    {
        int number_rows = nr;
        int number_cols = nc;
        int number_chans = nch;

        switch(process_dim)
        {
            case "col":
            {
                Matk matOut = new Matk(1, number_cols, number_chans);
                double[] vals = new double[number_rows];
                for(int k=0; k< number_chans; k++)
                    for (int j = 0; j < number_cols; j++)
                    {
                        for (int i = 0; i < number_rows; i++)
                            vals[i] = get(i, j, k);
                        matOut.set(functor.apply(vals),0,j,k);
                    }
                return matOut;
            }
            case "row":
            {
                Matk matOut = new Matk(number_rows, 1, number_chans);
                double[] vals = new double[number_cols];
                for(int k=0; k< number_chans; k++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        for (int j = 0; j < number_cols; j++)
                            vals[j] = get(i, j, k);
                        matOut.set(functor.apply(vals), i,0,k);
                    }
                return matOut;
            }
            case "channel":
            {
                Matk matOut = new Matk(number_rows, number_cols, 1);
                double[] vals = new double[number_chans];
                for (int j = 0; j < number_cols; j++)
                    for (int i = 0; i < number_rows; i++)
                    {
                        for(int k=0; k< number_chans; k++)
                            vals[k] = get(i, j, k);
                        matOut.set(functor.apply(vals), i,j,0);
                    }
                return matOut;
            }
            default:
                throw new IllegalArgumentException("ERROR: process_dim must be \"row\", \"col\" or \"channel\".");
        }
    }

    public Matk median(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.percentile(ee, 0.5));
    }

    public Matk percentile(String process_dim, double p)
    {
        return summarize(process_dim, ee->StatUtils.percentile(ee, p));
    }

    public Matk mode(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.mode(ee)[0]);
    }

    public Matk product(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.product(ee));
    }

    // sum all the elements in this matrix
    public double sum()
    {
        double total = 0;
        for(int ii=0; ii<ndata; ii++)
            total += data[ii];
        return total;
    }

    public Matk sum(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.sum(ee));
    }

    // Returns the sum of the natural logs of the entries in the input array, or Double.NaN if the array is empty.
    public Matk sumLog(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.sumLog(ee));
    }

    //Returns the sum of the squares of the entries in the input array, or Double.NaN if the array is empty.
    public Matk sumSq(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.sumSq(ee));
    }

    //Returns the variance of the entries in the input array, or Double.NaN if the array is empty.
    // population version = true means  ( sum((x_i - mean)^2) / n )
    // population version = false means  ( sum((x_i - mean)^2) / (n-1) )
    public Matk variance(String process_dim, boolean population_version)
    {
        if(population_version)
            return summarize(process_dim, ee->StatUtils.populationVariance(ee));
        else
            return summarize(process_dim, ee->StatUtils.variance(ee));
    }

    //Returns the variance of the entries in the input array, or Double.NaN if the array is empty.
    // ( sum((x_i - mean)^2) / (n-1) )
    public Matk variance(String process_dim)
    {
        return summarize(process_dim, ee->StatUtils.variance(ee));
    }

    //Returns the variance of the entries in the input array, using the precomputed mean value.
    // population version = true means  ( sum((x_i - mean)^2) / n )
    // population version = false means  ( sum((x_i - mean)^2) / (n-1) )
    public Matk variance(String process_dim, double mean, boolean population_version)
    {
        if(population_version)
            return summarize(process_dim, ee->StatUtils.populationVariance(ee, mean));
        else
            return summarize(process_dim, ee->StatUtils.variance(ee, mean));
    }

    //Returns the variance of the entries in the input array, using the precomputed mean value.
    //( sum((x_i - mean)^2) / (n-1) )
    public Matk variance(String process_dim, double mean)
    {
        return summarize(process_dim, ee->StatUtils.variance(ee, mean));
    }

    /**
     * compute histogram of this matrix; considers all the data in the whole matrix
     * gives same results as Matlab's histcounts/histc
     * @param edges
     * @return
     */
    public double[] hist(double[] edges)
    {
        if(!stdfuncs.is_sorted_ascend(edges))
            throw new IllegalArgumentException("ERROR: edges must be sorted in ascending in ascending order");

        int nedges = edges.length;
        int nbins = nedges - 1;
        double[] h = new double[nbins];

        int idx_edge_last = nedges - 1;
        double curVal;

        int idx_ub, idx_bin;

        int cc = 0;

        for(int k=0; k<nch; k++)
            for(int j=0; j<nc; j++)
                for(int i=0; i<nr; i++)
                {
                    curVal = data[cc++];
                    idx_ub = stdfuncs.bs_upper_bound(edges, curVal);

                    // handle boundary case (left most side)
                    if (idx_ub == 0)
                    {
                        // data less than e1 (the first edge), so don't count
                        if (curVal < edges[0])
                            continue;
                    }

                    // handle boundary case (right most side)
                    if (idx_ub == nedges)
                    {
                        // data greater than the last edge, so don't count
                        if (curVal > edges[idx_edge_last])
                            continue;
                        // need to decrement since due to being at exactly edge final
                        --idx_ub;
                    }

                    idx_bin = idx_ub - 1;
                    ++h[idx_bin];
                }

        return h;
    }

    // join this matrix with the given matrix matIn horizontally
    // if the number of rows or channels are different, max of them
    // will be taken and filled with zeros.
    public Matk add_cols(Matk matIn)
    {
        int nrows_new = Math.max(nr, matIn.nr);
        int ncols_new = nc + matIn.nc;
        int nch_new = Math.max(nch, matIn.nch);
        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        matOut.set(this, 0, nr-1, 0, nc-1, 0, nch-1);
        matOut.set(matIn, 0, matIn.nr-1, nc, ncols_new-1, 0, matIn.nch-1);
        return matOut;
    }

    // merge an array of matrices horizontally
    // if the number of rows or channels are different, max of them
    // will be taken and filled with zeros.
    public static Matk merge_cols(Matk[] vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.length;

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new = Math.max(nrows_new, vmat[kk].nr);
            ncols_new += vmat[kk].nc;
            nch_new = Math.max(nch_new, vmat[kk].nch);
        }

        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        int nc_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.set(vmat[kk], 0, vmat[kk].nr - 1, nc_count, nc_count + vmat[kk].nc - 1, 0, vmat[kk].nch - 1);
            nc_count += vmat[kk].nc;
        }

        return matOut;
    }

    // merge a list of matrices horizontally
    // if the number of rows or channels are different, max of them
    // will be taken and filled with zeros.
    public static Matk merge_cols(List<Matk> vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.size();

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new = Math.max(nrows_new, vmat.get(kk).nr);
            ncols_new += vmat.get(kk).nc;
            nch_new = Math.max(nch_new, vmat.get(kk).nch);
        }

        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        int nc_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.set(vmat.get(kk), 0, vmat.get(kk).nr - 1, nc_count, nc_count + vmat.get(kk).nc - 1, 0, vmat.get(kk).nch - 1);
            nc_count += vmat.get(kk).nc;
        }

        return matOut;
    }

    // join this matrix with the given matrix matIn vertically
    // if the number of cols or channels are different, max of them
    // will be taken and filled with zeros.
    public Matk add_rows(Matk matIn)
    {
        int nrows_new = nr + matIn.nr;
        int ncols_new = Math.max(nc, matIn.nc);
        int nch_new = Math.max(nch, matIn.nch);
        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        matOut.set(this, 0, nr - 1, 0, nc - 1, 0, nch - 1);
        matOut.set(matIn, nr, nrows_new - 1, 0, matIn.nc - 1, 0, matIn.nch - 1);
        return matOut;
    }

    // merge an array of matrices vertically
    // if the number of cols or channels are different, max of them
    // will be taken and filled with zeros.
    public static Matk merge_rows(Matk[] vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.length;

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new += vmat[kk].nr;
            ncols_new = Math.max(ncols_new, vmat[kk].nc);
            nch_new = Math.max(nch_new, vmat[kk].nch);
        }

        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        int nr_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.set(vmat[kk], nr_count, nr_count + vmat[kk].nr - 1, 0, vmat[kk].nc - 1, 0, vmat[kk].nch - 1);
            nr_count += vmat[kk].nr;
        }

        return matOut;
    }

    // merge a list of matrices vertically
    // if the number of cols or channels are different, max of them
    // will be taken and filled with zeros.
    public static Matk merge_rows(List<Matk> vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.size();

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new += vmat.get(kk).nr;
            ncols_new = Math.max(ncols_new, vmat.get(kk).nc);
            nch_new = Math.max(nch_new, vmat.get(kk).nch);
        }

        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        int nr_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.set(vmat.get(kk), nr_count, nr_count + vmat.get(kk).nr - 1, 0, vmat.get(kk).nc - 1, 0, vmat.get(kk).nch - 1);
            nr_count += vmat.get(kk).nr;
        }

        return matOut;
    }

    // add channels to the this matrix.
    // if rows and columns of the two matrices are different, max of them
    // will be taken and filled with zeros
    public Matk add_channels(Matk matIn)
    {
        int nrows_new = Math.max(nr, matIn.nr);
        int ncols_new = Math.max(nc, matIn.nc);
        int nch_new = nch + matIn.nch;
        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        matOut.set(this, 0, nr - 1, 0, nc - 1, 0, nch - 1);
        matOut.set(matIn, 0, matIn.nr - 1, 0, matIn.nc - 1, nch, nch_new - 1);
        return matOut;
    }

    // merge channels of an array of matrices
    // if rows and columns of the two matrices are different, max of them
    // will be taken and filled with zeros
    public static Matk merge_channels(Matk[] vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.length;

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new = Math.max(nrows_new, vmat[kk].nr);
            ncols_new = Math.max(ncols_new, vmat[kk].nc);
            nch_new += vmat[kk].nch;
        }

        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        int nch_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.set(vmat[kk], 0, vmat[kk].nr - 1, 0, vmat[kk].nc - 1, nch_count, nch_count + vmat[kk].nch - 1);
            nch_count += vmat[kk].nch;
        }

        return matOut;
    }

    // merge channels of a list of matrices
    // if rows and columns of the two matrices are different, max of them
    // will be taken and filled with zeros
    public static Matk merge_channels(List<Matk> vmat)
    {
        int nrows_new = 0;
        int ncols_new = 0;
        int nch_new = 0;
        int nmats = vmat.size();

        for (int kk = 0; kk < nmats; kk++)
        {
            nrows_new = Math.max(nrows_new, vmat.get(kk).nr);
            ncols_new = Math.max(ncols_new, vmat.get(kk).nc);
            nch_new += vmat.get(kk).nch;
        }

        Matk matOut = new Matk(nrows_new, ncols_new, nch_new);
        int nch_count = 0;

        for (int kk = 0; kk < nmats; kk++)
        {
            matOut.set(vmat.get(kk), 0, vmat.get(kk).nr - 1, 0, vmat.get(kk).nc - 1, nch_count, nch_count + vmat.get(kk).nch - 1);
            nch_count += vmat.get(kk).nch;
        }

        return matOut;
    }

    // remove cols from this matrix
    public Matk del_cols(int[] indices_remove)
    {
        // row indices that I want to keep (keep all; 1,2,...,nr)
        int[] row_idxs_keep = stdfuncs.fill_ladder_int(nr, 0, 1);

        // channel indices that I want to keep (keep all; 1,2,...,nch)
        int[] ch_idxs_keep = stdfuncs.fill_ladder_int(nch, 0, 1);

        // col indices to keep
        int[] col_idxs_all = stdfuncs.fill_ladder_int(nc, 0, 1);
        int[] col_idxs_keep = stdfuncs.set_diff(col_idxs_all, indices_remove);

        Matk matOut = get(row_idxs_keep, col_idxs_keep, ch_idxs_keep);
        return matOut;
    }

    // remove rows from this matrix
    public Matk del_rows(int[] indices_remove)
    {
        // col indices that I want to keep (keep all; 1,2,...,nr)
        int[] col_idxs_keep = stdfuncs.fill_ladder_int(nc, 0, 1);

        // channel indices that I want to keep (keep all; 1,2,...,nch)
        int[] ch_idxs_keep = stdfuncs.fill_ladder_int(nch, 0, 1);

        // row indices to keep
        int[] row_idxs_all = stdfuncs.fill_ladder_int(nr, 0, 1);
        int[] row_idxs_keep = stdfuncs.set_diff(row_idxs_all, indices_remove);

        Matk matOut = get(row_idxs_keep, col_idxs_keep, ch_idxs_keep);
        return matOut;
    }

    // remove channels from this matrix
    public Matk del_channels(int[] indices_remove)
    {
        // row indices that I want to keep (keep all; 1,2,...,nr)
        int[] row_idxs_keep = stdfuncs.fill_ladder_int(nr, 0, 1);

        // col indices that I want to keep (keep all; 1,2,...,nr)
        int[] col_idxs_keep = stdfuncs.fill_ladder_int(nc, 0, 1);

        // channel indices to keep
        int[] ch_idxs_all = stdfuncs.fill_ladder_int(nch, 0, 1);
        int[] ch_idxs_keep = stdfuncs.set_diff(ch_idxs_all, indices_remove);

        Matk matOut = get(row_idxs_keep, col_idxs_keep, ch_idxs_keep);
        return matOut;
    }

    // remove submatrix from this matrix
    public Matk del_submat(int[] row_indices_remove, int[] col_indices_remove, int[] channel_indices_remove)
    {
        // row indices to keep
        int[] row_idxs_all = stdfuncs.fill_ladder_int(nr, 0, 1);
        int[] row_idxs_keep = stdfuncs.set_diff(row_idxs_all, row_indices_remove);

        // col indices to keep
        int[] col_idxs_all = stdfuncs.fill_ladder_int(nc, 0, 1);
        int[] col_idxs_keep = stdfuncs.set_diff(col_idxs_all, col_indices_remove);

        // channel indices to keep
        int[] ch_idxs_all = stdfuncs.fill_ladder_int(nch, 0, 1);
        int[] ch_idxs_keep = stdfuncs.set_diff(ch_idxs_all, channel_indices_remove);

        Matk matOut = get(row_idxs_keep, col_idxs_keep, ch_idxs_keep);
        return matOut;
    }

    // find the locations of the elements in this matrix that satisfied
    // given number comparison condition
    public Result_find find(String comp_operator, double val)
    {
        int ini_capacity = Math.max(Math.max(nr, nc), nch);
        List<Integer> indices_list = new ArrayList<>(ini_capacity);
        List<Integer> i_list = new ArrayList<>(ini_capacity);
        List<Integer> j_list = new ArrayList<>(ini_capacity);
        List<Integer> k_list = new ArrayList<>(ini_capacity);
        List<Double> val_list = new ArrayList<>(ini_capacity);
        int cc = 0;
        double val_cur;

        switch(comp_operator)
        {
            case "=":
                for(int k=0; k<nch; k++)
                    for(int j=0; j<nc; j++)
                        for(int i=0; i<nr; i++)
                        {
                            val_cur = data[cc];
                            if(val_cur == val)
                            {
                                indices_list.add(cc);
                                i_list.add(i);
                                j_list.add(j);
                                k_list.add(k);
                                val_list.add(val_cur);
                            }
                            cc++;
                        }
                break;
            case ">=":
                for(int k=0; k<nch; k++)
                    for(int j=0; j<nc; j++)
                        for(int i=0; i<nr; i++)
                        {
                            val_cur = data[cc];
                            if(val_cur >= val)
                            {
                                indices_list.add(cc);
                                i_list.add(i);
                                j_list.add(j);
                                k_list.add(k);
                                val_list.add(val_cur);
                            }
                            cc++;
                        }
                break;
            case "<=":
                for(int k=0; k<nch; k++)
                    for(int j=0; j<nc; j++)
                        for(int i=0; i<nr; i++)
                        {
                            val_cur = data[cc];
                            if(val_cur <= val)
                            {
                                indices_list.add(cc);
                                i_list.add(i);
                                j_list.add(j);
                                k_list.add(k);
                                val_list.add(val_cur);
                            }
                            cc++;
                        }
                break;
            case ">":
                for(int k=0; k<nch; k++)
                    for(int j=0; j<nc; j++)
                        for(int i=0; i<nr; i++)
                        {
                            val_cur = data[cc];
                            if(val_cur > val)
                            {
                                indices_list.add(cc);
                                i_list.add(i);
                                j_list.add(j);
                                k_list.add(k);
                                val_list.add(val_cur);
                            }
                            cc++;
                        }
                break;
            case "<":
                for(int k=0; k<nch; k++)
                    for(int j=0; j<nc; j++)
                        for(int i=0; i<nr; i++)
                        {
                            val_cur = data[cc];
                            if(val_cur < val)
                            {
                                indices_list.add(cc);
                                i_list.add(i);
                                j_list.add(j);
                                k_list.add(k);
                                val_list.add(val_cur);
                            }
                            cc++;
                        }
                break;
            default:
                throw new IllegalArgumentException("ERROR: comp_operator must be \"=\", \">=\", \"<=\", \">\", \"<\"");
        }

        Result_find res = new Result_find();
        res.indices = stdfuncs.list_to_intArray(indices_list);
        res.iPos = stdfuncs.list_to_intArray(i_list);
        res.jPos = stdfuncs.list_to_intArray(j_list);
        res.kPos = stdfuncs.list_to_intArray(k_list);
        res.vals = stdfuncs.list_to_doubleArray(val_list);
        res.nFound = indices_list.size();

        return res;
    }

    // assume that this matrix is each data point stored as a col vector in a 2D matrix
    public Result_clustering kmeans_pp(int nclusters, int nMaxIters)
    {
        if(nch!=1)
            throw new IllegalArgumentException("ERROR: this matrix has more than one channel.");
        KMeansPlusPlusClusterer kmObj = new KMeansPlusPlusClusterer(nclusters, nMaxIters);
        List<vecPointForACMCluster_Matkc> tdata = new ArrayList<vecPointForACMCluster_Matkc>(nc);
        for(int j=0; j<nc; j++)
            tdata.add(new vecPointForACMCluster_Matkc(get_col(j)));
        List<CentroidCluster<vecPointForACMCluster_Matkc>> res = kmObj.cluster(tdata);

        Result_clustering res_cluster = new Result_clustering();

        res_cluster.centroids = new Matk(nr, nclusters);
        res_cluster.labels = new Matk(1, nc);
        res_cluster.nclusters = res.size();

        for(int j=0; j<res_cluster.nclusters; j++)
            res_cluster.centroids.set_col(res.get(j).getCenter().getPoint(), j);

        Matk dists = new Matk(1, res_cluster.nclusters);
        double dist;

        for(int j=0; j<nc; j++)
        {
            Matk cur_datapoint = get_col(j);
            // for this data point index j, compute euclidean distance to each of the
            // centroids and save it in dists
            for(int i=0; i<res_cluster.nclusters; i++)
            {
                dist = cur_datapoint.minus(res_cluster.centroids.get_col(i)).pow(2).sum();
                dists.set(dist,0,i,0);
            }

            // get the min distance
            Result_minmax res_min = dists.min();
            // record the cluster label for this data point index j
            res_cluster.labels.set(res_min.j + 1,0, j, 0);
        }

        return res_cluster;
    }

    /**
     * Generate a dataset that contains bivariate gaussian data for different number of classes
     * For each class, one multivariate gaussian distribution
     * @param nclasses
     * @param ndata_per_class Number of data for each class
     * @return an array of two matrices: (2xN) the generated data, (1xN) the labels
     * where N = ndata_per_class * nclasses
     */
    public static Result_labelled_data gen_dataset_BVN(int nclasses, int ndata_per_class)
    {
        double[] mean = new double[2];
        double[][] covariance = {{1,0}, {0, 1}};
        double[] sample;
        double rangeMin = -10;
        double rangeMax = 10;
        Random rand = new Random();

        Result_labelled_data res = new Result_labelled_data();
        res.dataset = new Matk(2, nclasses * ndata_per_class);
        res.labels = new Matk(1, nclasses * ndata_per_class);

        for(int i=0; i<nclasses; i++)
        {
            mean[0] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
            mean[1] = rangeMin + (rangeMax - rangeMin) * rand.nextDouble();
            MultivariateNormalDistribution dist = new MultivariateNormalDistribution(mean, covariance);
            int idx_col;
            for(int j=0; j<ndata_per_class; j++)
            {
                idx_col = (i*ndata_per_class)+j;
                res.dataset.set_col(dist.sample(), idx_col);
                res.labels.set(i+1, idx_col);
            }
        }

        return res;
    }

    // 0.2989 * R + 0.5870 * G + 0.1140 * B
    public Matk rgb2gray()
    {
        if (nch != 3)
            throw new IllegalArgumentException("ERROR: The input matrix must have 3 channels.");

        Matk mOut = new Matk(nr, nc, 1);
        double[] ptr_out = mOut.data;

        for (int ii = 0; ii < ndata_per_chan; ii++)
            ptr_out[ii] = 0.2989 * data[ii] + 0.5870 * data[ndata_per_chan + ii] + 0.1140 * data[ndata_per_chan * 2 + ii];

        return mOut;
    }

    // 0.2989 * R + 0.5870 * G + 0.1140 * B
    public Matk bgr2gray()
    {
        if (nch != 3)
            throw new IllegalArgumentException("ERROR: The input matrix must have 3 channels.");

        Matk mOut = new Matk(nr, nc, 1);
        double[] ptr_out = mOut.data;

        for (int ii = 0; ii < ndata_per_chan; ii++)
            ptr_out[ii] = 0.1140 * data[ii] + 0.5870 * data[ndata_per_chan + ii] +  0.2989 * data[ndata_per_chan * 2 + ii];

        return mOut;
    }

    // normalize dataset using pnorm
    // treat each col of the matrix of a data point
    // for each data point (vector), divide all the elements
    // of the vector by SUM(ABS(V).^P)^(1/P)
    // Note: modify this matrix, just return a reference
    public Matk normalize_dataset_pNorm(double p)
    {
        if(nch != 1)
            throw new IllegalArgumentException("ERROR: The number of channels of this matrix must be one.");

        double s;
        double tol = 0.00001; // just in case of division by zero
        double p_inv = 1.0 / p;
        double[] v;

        for(int j=0; j<nc; j++)
        {
            v = get_col_arrayOutput(j);
            s = 0;
            for(int i=0; i<nr; i++)
                s += (Math.pow(Math.abs(v[i]), p));
            s = Math.pow(s, p_inv);
            for(int i=0; i<nr; i++)
                v[i] = v[i] / (s + tol);
            set_col(v, j);
        }
        return this;
    }

    public Matk normalize_dataset_L2Norm()
    {
        return normalize_dataset_pNorm(2.0);
    }

    public Matk normalize_dataset_L1Norm()
    {
        return normalize_dataset_pNorm(1.0);
    }

}


class vecPointForACMCluster_Matkc implements Clusterable
{
    double[] data;

    vecPointForACMCluster_Matkc(Matk m)
    {
        data = m.vectorize_to_doubleArray();
    }

    @Override
    public double[] getPoint() {
        return data;
    }

}