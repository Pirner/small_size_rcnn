import tensorflow as tf


class RCNNTrainWrapper:
    def __init__(self):
        self._model = None
        self._optimizer = None
        self._loss = None
        self._metrics = None

        self._build_vgg16_model()
        self._build_optimizer()

    def _compile_model(self):
        """
        compile built together stuff
        :return:
        """
        self._loss = tf.keras.losses.categorical_crossentropy
        self._metrics = ['accuracy']

        self._model.compile(loss=self._loss, optimizer=self._optimizer, metrics=self._metrics)

    def _build_optimizer(self):
        """
        builds optimizer to train networks with
        :return:
        """
        self._optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    def _build_vgg16_model(self):
        """
        builds rcnn model with vgg16 as backbone (legacy implementation)
        :return:
        """
        vgg_model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)
        # vgg_model.summary()

        for layers in vgg_model.layers[:15]:
            layers.trainable = False

        x = vgg_model.layers[-2].output

        predictions = tf.keras.layers.Dense(2, activation='softmax')(x)

        model_final = tf.keras.Model(inputs=vgg_model.input, outputs=predictions)

        model_final.summary()
        self._model = model_final
