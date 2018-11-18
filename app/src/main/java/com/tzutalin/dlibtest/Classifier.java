package com.tzutalin.dlibtest;

import android.app.Activity;
import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.os.SystemClock;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class Classifier {
    private static final String LOG_TAG = Classifier.class.getSimpleName();

    private static final String MODEL_PATH = "converted_model.tflite";

    private static final int DIM_BATCH_SIZE = 11;
    private static final int CATEGORY_COUNT = 4;

    private final Interpreter mInterpreter;
    private final ByteBuffer mImgData;
    private final float[][] mResult = new float[1][CATEGORY_COUNT];

    public Classifier(Context context) throws IOException {
        mInterpreter = new Interpreter(loadModelFile(context));
        mImgData = ByteBuffer.allocateDirect(4 * 2 * DIM_BATCH_SIZE);
        mImgData.order(ByteOrder.nativeOrder());
    }

    public Result classify(float[] input) {
        mImgData.rewind();
        for(float flt: input) {
            mImgData.putFloat(flt);
        }
        long startTime = SystemClock.uptimeMillis();
        mInterpreter.run(mImgData, mResult);
        long endTime = SystemClock.uptimeMillis();
        long timeCost = endTime - startTime;
        return new Result(mResult[0], timeCost);
    }

    private MappedByteBuffer loadModelFile(Context context) throws IOException {
        AssetFileDescriptor fileDescriptor = context.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

}
