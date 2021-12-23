package com.trpoal.photooptimizer.helpers;

import android.graphics.Bitmap;

public class Results {
    public Bitmap bitmap;
    public String title;

    public Results(Bitmap bitmap, String title) {
        this.bitmap = bitmap;
        this.title = title;
    }
}
