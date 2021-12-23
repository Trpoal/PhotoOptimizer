package com.trpoal.photooptimizer.helpers;

import android.app.Activity;

import org.tensorflow.lite.support.common.TensorOperator;
import org.tensorflow.lite.support.common.ops.NormalizeOp;

import java.io.IOException;

public class MyClassifier extends Classifier {
        private static final float IMAGE_MEAN = 0.0f;

        private static final float IMAGE_STD = 255.0f;

        private static final float PROBABILITY_MEAN = 0.0f;

        private static final float PROBABILITY_STD = 1.0f;

        public MyClassifier(Activity activity, Device device, int numThreads)
                throws IOException {
            super(activity, device, numThreads);
        }

        @Override
        protected String getModelPath() {
            return "model5.tflite";
        }

        @Override
        protected String getLabelPath() {
            return "labels.txt";
        }

        @Override
        protected TensorOperator getPreprocessNormalizeOp() {
            return new NormalizeOp(IMAGE_MEAN, IMAGE_STD);
        }

        @Override
        protected TensorOperator getPostprocessNormalizeOp() {
            return new NormalizeOp(PROBABILITY_MEAN, PROBABILITY_STD);
        }
}
