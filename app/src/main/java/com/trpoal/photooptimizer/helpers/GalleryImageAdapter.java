package com.trpoal.photooptimizer.helpers;

import android.content.Context;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Gallery;
import android.widget.ImageView;

import java.util.ArrayList;

public class GalleryImageAdapter extends BaseAdapter
{
    private Context mContext;
    public ArrayList<Results> List = new ArrayList<Results>();

    public GalleryImageAdapter(Context context)
    {
        mContext = context;
    }

    public int getCount() {
        return List.size();
    }

    public Object getItem(int position) {
        return position;
    }

    public long getItemId(int position) {
        return position;
    }

    public View getView(int index, View view, ViewGroup viewGroup)
    {
        ImageView i = new ImageView(mContext);

        i.setImageBitmap(List.get(index).bitmap);
        i.setLayoutParams(new Gallery.LayoutParams(250, 250));

        i.setScaleType(ImageView.ScaleType.FIT_XY);
        return i;
    }
}
