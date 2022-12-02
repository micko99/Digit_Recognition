from flask import Flask, render_template, send_from_directory, url_for, redirect
from flask_uploads import IMAGES, UploadSet, configure_uploads
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import SubmitField
from flask_wtf import FlaskForm

from AI_final import load_image_and_return_number
from run_program import run_app
import os

app = Flask(__name__)

app.config['SECRET_KEY']= 'secretkey'
app.config['UPLOADED_PHOTOS_DEST'] = 'uploads'

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)

class UploadForm(FlaskForm):
    image = FileField(
        validators = [
            FileAllowed(photos, 'Only images are allowed'),
            FileRequired('File field should not be empty')
        ]
    )
    submit = SubmitField('Upload')
    submit_2 = SubmitField("Show result")


@app.route('/uploads/<filename>')
def get_file(filename):
    return send_from_directory(app.config['UPLOADED_PHOTOS_DEST'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    form = UploadForm()
    if form.validate_on_submit():
        filename = photos.save(form.image.data)
        file_url = url_for('get_file', filename=filename)
        result = load_image_and_return_number(file_url[1:]) # this url is /uploads/{image}
        os.remove(file_url[1:])
    else:
        file_url = None
        result = None
    return render_template('index.html', form=form, file_url=file_url, result=result)

if __name__=='__main__':
    run_app()