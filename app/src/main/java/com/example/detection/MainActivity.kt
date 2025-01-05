package com.example.detection

import android.content.ContentValues.TAG
import android.content.Intent
import android.graphics.Bitmap
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Base64
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import android.util.Log
import java.io.ByteArrayOutputStream

class MainActivity : AppCompatActivity() {

    lateinit var imageView: ImageView
    lateinit var bitmap: Bitmap


    val python = Python.getInstance()
    val pythonModule = python.getModule("number_detection")

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView2)
        var textView: TextView = findViewById(R.id.textView)

        var select: Button = findViewById(R.id.select)
        select.setOnClickListener(View.OnClickListener {
            var intent: Intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type ="image/*"

            startActivityForResult(intent, 100)

        })




        var predict: Button = findViewById(R.id.predict)
        predict.setOnClickListener(View.OnClickListener {
            var baos: ByteArrayOutputStream = ByteArrayOutputStream ()

            bitmap.compress(Bitmap.CompressFormat.PNG, 100, baos)

            var imageBytes : ByteArray = baos.toByteArray()

            var encodedImage : String = android.util.Base64.encodeToString(imageBytes, Base64.DEFAULT)

            //Log.d("EXAMPLE", encodedImage)

            val result: PyObject = pythonModule.callAttr("prediction_numbers", encodedImage)

            val result_experimental: PyObject = pythonModule.callAttr("prediction_numbers_experimental", encodedImage)

            //val result: PyObject = pythonModule.callAttr("prediction_numbers", "out.png")
            val message: String = "Result of prediction: " + result.toString() + "\n" + "Result of prediction (experimental): " + result_experimental.toString()
            textView.setText(message)
        })
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        imageView.setImageURI(data?.data)

        var uri: Uri? = data?.data
        bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
    }

}

