import 'dart:io';
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() => runApp(const MangoDiseaseApp());

class MangoDiseaseApp extends StatelessWidget {
  const MangoDiseaseApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: HomePage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  Interpreter? _interpreter;
  List<String> _labels = [];
  File? _imageFile;
  String _prediction = '';
  double _confidence = 0.0;

  static const int inputSize = 224;

  @override
  void initState() {
    super.initState();
    _loadModelAndLabels();
  }

  Future<void> _loadModelAndLabels() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/models/model_unquant.tflite');
      String labelsData = await rootBundle.loadString('assets/models/labels.txt');
      _labels = labelsData
          .split('\n')
          .map((s) => s.trim())
          .where((s) => s.isNotEmpty)
          .toList();

      print('Model and labels loaded: ${_labels.length} classes');
    } catch (e) {
      print('Error loading model or labels: $e');
    }
  }

  Future<void> _pickImage(ImageSource source) async {
    final picker = ImagePicker();
    final picked = await picker.pickImage(source: source);
    if (picked != null) {
      setState(() {
        _imageFile = File(picked.path);
      });
      await _runModelOnImage(_imageFile!);
    }
  }

  Future<ui.Image> _decodeAndResize(File file) async {
    final bytes = await file.readAsBytes();
    final codec = await ui.instantiateImageCodec(bytes,
        targetWidth: inputSize, targetHeight: inputSize);
    final frame = await codec.getNextFrame();
    return frame.image;
  }

  Future<Float32List> _imageToInputTensor(ui.Image image) async {
    final byteData = await image.toByteData(format: ui.ImageByteFormat.rawRgba);
    if (byteData == null) {
      throw Exception("Failed to get byte data");
    }
    final bytes = byteData.buffer.asUint8List();
    final input = Float32List(inputSize * inputSize * 3);

    int j = 0;
    for (int i = 0; i < bytes.length; i += 4) {
      final r = bytes[i];
      final g = bytes[i + 1];
      final b = bytes[i + 2];
      input[j++] = r / 255.0; // normalize to [0,1]
      input[j++] = g / 255.0;
      input[j++] = b / 255.0;
    }
    return input;
  }
Future<void> _runModelOnImage(File imageFile) async {
  if (_interpreter == null) {
    print('Interpreter not loaded');
    return;
  }

  print('Running model on image: ${imageFile.path}');

  final image = await _decodeAndResize(imageFile);
  final input = await _imageToInputTensor(image);

  var inputTensor = input.reshape([1, inputSize, inputSize, 3]);
  var output = [List.filled(_labels.length, 0.0)];  // 👈 FIXED

  _interpreter!.run(inputTensor, output);

  print('Model output: $output');

  final scores = output[0];  // 👈 Access inner list
  int maxIndex = 0;
  double maxScore = scores[0];

  for (int i = 1; i < scores.length; i++) {
    if (scores[i] > maxScore) {
      maxScore = scores[i];
      maxIndex = i;
    }
  }

  final detectedLabel = _labels[maxIndex];
  final detectedScore = maxScore;

  print('👉 Detected: $detectedLabel with confidence ${(detectedScore * 100).toStringAsFixed(1)}%');

  setState(() {
    _prediction = detectedLabel;
    _confidence = detectedScore;
  });
}


  @override
  void dispose() {
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Mango Disease Detector'),
      ),
      body: Padding(
        padding: const EdgeInsets.fromLTRB(15, 15, 15, 40),
        child: Column(
          children: [
            _imageFile == null
                ? const Expanded(
                  flex: 8,
                  child: Center(child: Icon(Icons.image, size: 300, color: Colors.grey,)),
                )
                : Image.file(_imageFile!),
            const SizedBox(height: 100),
            Expanded(
              flex: 3,
              child: _prediction.isEmpty
                  ? const Text(
                      '',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 18),
                    )
                  : RichText(
                      textAlign: TextAlign.center,
                      text: TextSpan(
                        style: const TextStyle(fontSize: 18, color: Colors.black),
                        children: [
                          const TextSpan(
                            text: 'Prediction: ',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          TextSpan(
                            text: _prediction,
                            style: TextStyle(
                              color: _prediction == 'Healthy' ? Colors.green : Colors.black,
                              fontWeight: FontWeight.normal,
                            ),
                          ),
                          const TextSpan(
                            text: '\nConfidence: ',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          TextSpan(
                            text: '${(_confidence * 100).toStringAsFixed(1)}%',
                            style: const TextStyle(fontWeight: FontWeight.normal),
                          ),
                        ],
                      ),
                    ),
            ),

            Expanded(
              flex: 1,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  IconButton(onPressed: () => _pickImage(ImageSource.gallery), icon: const Icon(Icons.image, size: 40)),
                  IconButton(onPressed: () => _pickImage(ImageSource.camera), icon: const Icon(Icons.camera, size: 40)),
                ],
              ),
            )
          ],
        ),
      ),
    );
  }
}
