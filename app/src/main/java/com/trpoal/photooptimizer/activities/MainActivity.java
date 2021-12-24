package com.trpoal.photooptimizer.activities;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.provider.MediaStore;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Surface;
import android.view.View;
import android.widget.Gallery;
import android.widget.ProgressBar;
import android.widget.TextView;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.FileProvider;

import com.trpoal.photooptimizer.R;
import com.trpoal.photooptimizer.helpers.GalleryImageAdapter;
import com.trpoal.photooptimizer.views.ImageSource;
import com.trpoal.photooptimizer.helpers.Results;
import com.trpoal.photooptimizer.views.SubsamplingScaleImageView;
import com.trpoal.photooptimizer.helpers.Classifier;

import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {
    private static int REQUEST_IMAGE_CAPTURE = 1;
    private final int PICK_IMAGE = 2;
    private static final int PERMISSIONS_REQUEST = 1;
    private static final String PERMISSION_CAMERA = Manifest.permission.CAMERA;
    private static final String PERMISSION_STORAGE_READ = Manifest.permission.READ_EXTERNAL_STORAGE;
    private static final String PERMISSION_STORAGE_WRITE = Manifest.permission.WRITE_EXTERNAL_STORAGE;

    private int imageSizeX;
    private int imageSizeY;

    private String currentPhotoPath;
    private Classifier classifier;
    private HandlerThread handlerThread;
    private TextView textView;
    private File photoFile;
    private ProgressBar progressBar;
    private Bitmap finalBitmap;
    private Handler handler;

    private SubsamplingScaleImageView selectedImage;
    private GalleryImageAdapter galleryImageAdapter;

    @SuppressLint("ResourceAsColor")
    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        textView = findViewById(R.id.text);
        Toolbar myToolbar = findViewById(R.id.toolbar4);
        myToolbar.setTitleTextColor(getColor(R.color.text_color_toolbar));
        setSupportActionBar(myToolbar);

        Gallery gallery = findViewById(R.id.gallery);
        selectedImage= findViewById(R.id.image);
        gallery.setSpacing(1);
        galleryImageAdapter= new GalleryImageAdapter(this);
        gallery.setAdapter(galleryImageAdapter);

        progressBar = findViewById(R.id.progress_circular);

        gallery.setOnItemClickListener((parent, v, position, id) -> {
            selectedImage.setImage(ImageSource.bitmap(galleryImageAdapter.List.get(position).bitmap));
            textView.setText(String.format("Result: %s", galleryImageAdapter.List.get(position).title));
        });
        recreateClassifier(Classifier.Model.MyModel, Classifier.Device.GPU, -1);

        handlerThread = new HandlerThread("inference");
        handlerThread.start();
        handler = new Handler(handlerThread.getLooper());
        if (!hasPermission()) {
            requestPermission();
        }
        if (!OpenCVLoader.initDebug())
            Log.d("ERROR", "Unable to load OpenCV");
        else
            Log.d("SUCCESS", "OpenCV loaded");
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main_activity_actions, menu);
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_open_folder:
                reset();
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE);
                break;
            default:
                return super.onOptionsItemSelected(item);
        }
        return true;
    }

    private void reset() {
        galleryImageAdapter.List.clear();
        galleryImageAdapter.notifyDataSetChanged();
        selectedImage.reset(false);
    }

    private boolean hasPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            return checkSelfPermission(PERMISSION_CAMERA) == PackageManager.PERMISSION_GRANTED
                    && checkSelfPermission(PERMISSION_STORAGE_READ) == PackageManager.PERMISSION_GRANTED
                    && checkSelfPermission(PERMISSION_STORAGE_WRITE) == PackageManager.PERMISSION_GRANTED;
        } else {
            return true;
        }
    }

    private void showProgressBar() {
        progressBar.setVisibility(View.VISIBLE);
    }

    private void hideProgressBar() {
        progressBar.setVisibility(View.GONE);
    }

    private void startOptimizer() {
        ByteArrayOutputStream stream = new ByteArrayOutputStream();
        finalBitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
        byte[] byteArray = stream.toByteArray();

        Mat srcImage = Imgcodecs.imdecode(new MatOfByte(byteArray), Imgcodecs.IMREAD_UNCHANGED);
        Mat grayImage = new Mat();
        Imgproc.cvtColor(srcImage, grayImage, Imgproc.COLOR_RGB2GRAY);

        MatOfInt historamSize = new MatOfInt(256);
        MatOfFloat histogramRange = new MatOfFloat(0f, 256f);
        MatOfInt chanel = new MatOfInt(0);
        Mat mask = new Mat();
        Mat histogram = new Mat();

        List<Mat> list = new ArrayList<Mat>();
        list.add(grayImage);
        Imgproc.calcHist(list, chanel, mask, histogram, historamSize, histogramRange);

        grayImage.release();
        mask.release();

        ArrayList<Float> accumulator = new ArrayList<Float>();
        accumulator.add((float)histogram.get(0, 0)[0]);
        for (int i = 1; i < 256; i++)
        {
            accumulator.add(accumulator.get(i - 1) + (float)histogram.get(i, 0)[0]);
        }

        histogram.release();

        Float maximum = accumulator.get(256 - 1);

        double clipHistPercent = 1.0;
        clipHistPercent *= (maximum / 100.0);
        clipHistPercent /= 2.0;

        int minimumGray = 0;
        while (accumulator.get(minimumGray) < clipHistPercent)
        {
            minimumGray++;
        }

        int maximumGray = 256 - 1;
        while (accumulator.get(maximumGray) >= (maximum - clipHistPercent))
        {
            maximumGray--;
        }

        double alpha = 255.0 / (maximumGray - minimumGray);
        double beta = -minimumGray * alpha;
        srcImage.convertTo(srcImage, -1, alpha, beta);

        MatOfByte resultImage = new MatOfByte();
        Imgcodecs.imencode(".jpg", srcImage, resultImage);

        byte[] result = resultImage.toArray();
        Bitmap bitmap = BitmapFactory.decodeByteArray(result, 0, result.length);

        runOnUiThread(() -> {
            hideProgressBar();
            Results results = new Results(bitmap, "Optimized");
            galleryImageAdapter.List.add(results);
            selectedImage.setImage(ImageSource.bitmap(results.bitmap));
            galleryImageAdapter.notifyDataSetChanged();
            textView.setText(String.format("Result: %s", results.title));
        });
    }

    public void onFloatButtonClick(View view) {
        reset();
        try {
            photoFile = CreateImageFile();
        } catch (Exception ex) {
            int aa = 5;
        }
        if (photoFile != null) {
            Uri imageUri = FileProvider.getUriForFile(this, "com.trpoal.photooptimizer.provider", photoFile);
            Intent intent = new Intent((MediaStore.ACTION_IMAGE_CAPTURE));
            intent.putExtra(MediaStore.EXTRA_OUTPUT, imageUri);
            startActivityForResult(intent, REQUEST_IMAGE_CAPTURE);
        }
    }

    @Override
    public void onRequestPermissionsResult(
            final int requestCode, final String[] permissions, final int[] grantResults) {
        if (requestCode == PERMISSIONS_REQUEST) {
            if (!allPermissionsGranted(grantResults)) {
                requestPermission();
            }
        }
    }

    private static boolean allPermissionsGranted(final int[] grantResults) {
        for (int result : grantResults) {
            if (result != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private void requestPermission() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
            requestPermissions(new String[] {PERMISSION_CAMERA, PERMISSION_STORAGE_READ, PERMISSION_STORAGE_WRITE}, PERMISSIONS_REQUEST);
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        showProgressBar();
        runInBackground(() -> {
            Bitmap bitmap = null;

            if (requestCode == REQUEST_IMAGE_CAPTURE && resultCode == RESULT_OK) {
                bitmap = BitmapFactory.decodeFile(photoFile.getPath());
            } else if (requestCode == PICK_IMAGE && resultCode == RESULT_OK) {
                bitmap = getFolderByPath(data);
            }

            finalBitmap = bitmap;
            runOnUiThread(() -> {
                if (finalBitmap != null) {
                    galleryImageAdapter.List.clear();
                    galleryImageAdapter.List.add(new Results(finalBitmap, "Original image"));
                    galleryImageAdapter.notifyDataSetChanged();
                    runInBackground(() -> processImage(finalBitmap));
                }
                else {
                    hideProgressBar();
                }
            });
        });
    }

    private Bitmap getFolderByPath(@Nullable Intent data) {
        Bitmap bitmap = null;
        if (data != null) {
            Uri uri = data.getData();
            try {
                bitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                photoFile = CreateImageFile();
                OutputStream stream = new FileOutputStream(photoFile);
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, stream);
                stream.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
        return bitmap;
    }

    private void recreateClassifier(Classifier.Model model, Classifier.Device device, int numThreads) {
        if (classifier != null) {
            classifier.close();
            classifier = null;
        }
        try {
            classifier = Classifier.create(this, model, device, numThreads);
        } catch (IOException e) {
            Log.d("Exception", e.getMessage());
        }

        imageSizeX = classifier.getImageSizeX();
        imageSizeY = classifier.getImageSizeY();
    }

    protected synchronized void runInBackground(final Runnable r) {
        if (handler != null) {
            handler.post(r);
        }
    }

    protected int getScreenOrientation() {
        switch (getWindowManager().getDefaultDisplay().getRotation()) {
            case Surface.ROTATION_270:
                return 270;
            case Surface.ROTATION_180:
                return 180;
            case Surface.ROTATION_90:
                return 90;
            default:
                return 0;
        }
    }

    @SuppressLint("ResourceAsColor")
    @RequiresApi(api = Build.VERSION_CODES.N)
    private void processImage(Bitmap bit) {
        if (classifier != null) {
            ArrayList<Bitmap> bitmaps = createBitmaps(bit);
            int blurCount = 0;
            for (int i = 0; i < bitmaps.size(); i++) {
                List<Classifier.Recognition> list = classifier.recognizeImage(bitmaps.get(i), getScreenOrientation());
                galleryImageAdapter.List.add(new Results(bitmaps.get(i), list.get(0).getTitle()));
                if (list.get(0).getTitle().equals("Not Clear")) {
                    blurCount++;
                }
            }

            processResults(blurCount, bitmaps.size());
        }
    }

    private void processResults(int finalBlurCount, int bitmapsSize) {
        runOnUiThread(() -> {
            if (finalBlurCount >= bitmapsSize / 2) {
                new AlertDialog.Builder(this)
                        .setTitle("Внимание!")
                        .setMessage("Картинка смазана!")
                        .setPositiveButton("Улучшить", (dialog, id) ->
                        {
                            dialog.dismiss();
                            showProgressBar();
                            runInBackground(() -> startOptimizer());
                        })
                        .setNeutralButton("Оставить", (dialog, id) -> {
                        })
                        .create()
                        .show();
            }
            galleryImageAdapter.notifyDataSetChanged();
            hideProgressBar();
        });
    }

    @RequiresApi(api = Build.VERSION_CODES.N)
    public ArrayList<Bitmap> createBitmaps(Bitmap source){
        ArrayList<Bitmap> bmp = new ArrayList<>();
        int size = 600;
        int width = source.getWidth();
        int height = source.getHeight();
        int vertCount = height / size;
        int vertMargin = height - vertCount * size;
        int horCount = width / size;
        int hotMargin = width - horCount*size;
        for (int x = vertMargin/2; x < height - vertMargin/2; x=x+size)
        {
            for (int y = hotMargin/2; y < width - hotMargin/2; y=y+size)
            {
                bmp.add(Bitmap.createBitmap(source, y, x, size, size));
            }
        }
        return bmp;
    }

    private File CreateImageFile() throws IOException {
        String imageFileName = "fileName";
        File storageDir = getExternalFilesDir(Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(imageFileName, ".jpg", storageDir);
        currentPhotoPath = image.getAbsolutePath();
        return image;
    }
}
