package com.trpoal.photooptimizer.activities;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.AlertDialog;
import android.content.Context;
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
import android.os.SystemClock;
import android.provider.MediaStore;
import android.view.Menu;
import android.view.MenuItem;
import android.view.Surface;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.Gallery;
import android.widget.ImageView;
import android.widget.ProgressBar;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.content.FileProvider;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.trpoal.photooptimizer.R;
import com.trpoal.photooptimizer.helpers.ImageSource;
import com.trpoal.photooptimizer.helpers.SubsamplingScaleImageView;
import com.trpoal.photooptimizer.helpers.Classifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.UUID;

import uk.co.samuelwall.materialtaptargetprompt.MaterialTapTargetPrompt;
import uk.co.samuelwall.materialtaptargetprompt.extras.PromptBackground;
import uk.co.samuelwall.materialtaptargetprompt.extras.backgrounds.RectanglePromptBackground;
import uk.co.samuelwall.materialtaptargetprompt.extras.focals.RectanglePromptFocal;

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

    private MenuItem approveItem;
    private MenuItem declineItem;

    private SubsamplingScaleImageView selectedImage;
    private GalleryImageAdapter galleryImageAdapter;

    String finalResult = "Processing... ";

    @SuppressLint("ResourceAsColor")
    @RequiresApi(api = Build.VERSION_CODES.M)
    @Override

    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        File clear = new File(Environment.getExternalStorageDirectory().getAbsolutePath()+"/clear/");
        clear.mkdirs();
        File notClear = new File(Environment.getExternalStorageDirectory().getAbsolutePath()+"/notclear/");
        notClear.mkdirs();
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
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.main_activity_actions, menu);
        approveItem = menu.findItem(R.id.action_approve);
        declineItem = menu.findItem(R.id.action_decline);
        approveItem.setVisible(false);
        declineItem.setVisible(false);
        return super.onCreateOptionsMenu(menu);
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item)
    {
        String result = textView.getText().toString();
        switch (item.getItemId())
        {
            case R.id.action_approve:
                if (result.contains("Clear"))
                {
                    try {
                        moveFileToFolder("clear");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                else
                {
                    try {
                        moveFileToFolder("notclear");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;
            case R.id.action_decline:
                if (result.contains("Clear"))
                {
                    try {
                        moveFileToFolder("notclear");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                else
                {
                    try {
                        moveFileToFolder("clear");
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                break;
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

    private void reset()
    {
        declineItem.setVisible(false);
        approveItem.setVisible(false);
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

    private void showProgressBar()
    {
        progressBar.setVisibility(View.VISIBLE);
    }

    private void hideProgressBar()
    {
        progressBar.setVisibility(View.GONE);
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

    private void startOptimizer()
    {

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
            Bitmap finalBitmap = bitmap;
            runOnUiThread(() -> {
                if (finalBitmap != null) {
                    declineItem.setVisible(true);
                    approveItem.setVisible(true);
                    galleryImageAdapter.List.clear();
                    galleryImageAdapter.List.add(new Results(finalBitmap, finalResult));
                    galleryImageAdapter.notifyDataSetChanged();
                    processImage(finalBitmap);
                }
                else
                {
                    hideProgressBar();
                }
            });
        });
    }

    private void recreateClassifier(Classifier.Model model, Classifier.Device device, int numThreads) {
        if (classifier != null) {
            classifier.close();
            classifier = null;
        }
        try {
            classifier = Classifier.create(this, model, device, numThreads);
        } catch (IOException e) {

        }

        imageSizeX = classifier.getImageSizeX();
        imageSizeY = classifier.getImageSizeY();
    }

    private Handler handler;

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

    public class Results
    {
        public Bitmap bitmap;
        public String title;

        public Results(Bitmap bitmap, String title)
        {
            this.bitmap = bitmap;
            this.title = title;
        }
    }

    @SuppressLint("ResourceAsColor")
    @RequiresApi(api = Build.VERSION_CODES.N)
    protected void processImage(Bitmap bit) {
        runInBackground(
                () -> {
                    if (classifier != null) {
                        final long startTime = SystemClock.uptimeMillis();

                        ArrayList<Bitmap> bitmaps = createBitmaps(bit);
                        int blurCount = 0;
                        for (int i = 0; i < bitmaps.size(); i++) {
                            List<Classifier.Recognition> list = classifier.recognizeImage(bitmaps.get(i), getScreenOrientation());
                            galleryImageAdapter.List.add(new Results(bitmaps.get(i), list.get(0).getTitle()));
                            if (list.get(0).getTitle().equals("Not Clear")) {
                                blurCount++;
                            }
                        }

                        int finalBlurCount = blurCount;
                        runOnUiThread(
                                () -> {
                                    if (finalBlurCount >= bitmaps.size() / 2) {
                                        finalResult = "Not Clear";
                                        new AlertDialog.Builder(this)
                                                .setTitle("Внимание!")
                                                .setMessage("Картинка смазана!")
                                                .setPositiveButton("Переделать", (dialog, id) -> startOptimizer())
                                                .setNeutralButton("Продолжить", (dialog, id) -> {
                                                })
                                                .create()
                                                .show();
                                    }
                                    galleryImageAdapter.notifyDataSetChanged();
                                    hideProgressBar();
                                });
                    }
                });
    }

    @SuppressLint("ResourceAsColor")
    private void shopApprovePrompt() {
        new MaterialTapTargetPrompt.Builder(MainActivity.this)
                .setTarget(R.id.action_approve)
                .setCaptureTouchEventOnFocal(false)
                .setBackgroundColour(R.color.semi_transparent2)
                .setIcon(R.drawable.outline_done_black_18dp)
                .setCaptureTouchEventOutsidePrompt(false)
                .setAutoDismiss(false)
                .setPrimaryText("Approve it!!!")
                .setSecondaryText("APPPPPROVE!")
                .setPromptStateChangeListener((prompt, state) -> {
                    if (state == MaterialTapTargetPrompt.STATE_DISMISSED)
                    {
                        // User has pressed the prompt target
                    }
                })
                .show();
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

    private void moveFileToFolder(String folderName) throws IOException {
        if (photoFile != null) {
            File to = new File(Environment.getExternalStorageDirectory()
                    .getAbsolutePath()+"/"+folderName+"/"+ UUID.randomUUID() +".jpg");
            if (!to.exists())
            {
                to.createNewFile();
            }
            photoFile.renameTo(to);
            Toast.makeText(this, "ADDED TO " + folderName, Toast.LENGTH_LONG).show();
            declineItem.setVisible(false);
            approveItem.setVisible(false);
        }
    }
}
