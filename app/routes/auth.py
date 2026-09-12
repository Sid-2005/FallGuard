from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from app.models.database import db
from app.models.user import User

auth_bp = Blueprint('auth', __name__)

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.monitor'))
    if request.method == 'POST':
        identifier = request.form.get('identifier', '').strip()
        password   = request.form.get('password', '')
        user = User.query.filter(
            (User.username == identifier) | (User.email == identifier)
        ).first()
        if user and user.is_active and check_password_hash(user.password_hash, password):
            login_user(user, remember=bool(request.form.get('remember')))
            user.last_login = datetime.utcnow()
            db.session.commit()
            return redirect(request.args.get('next') or url_for('dashboard.monitor'))
        flash('Invalid credentials.', 'error')
    return render_template('pages/login.html')

@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard.monitor'))
    if request.method == 'POST':
        username  = request.form.get('username','').strip()
        email     = request.form.get('email','').strip()
        full_name = request.form.get('full_name','').strip()
        password  = request.form.get('password','')
        if request.form.get('confirm_password') != password:
            flash('Passwords do not match.', 'error')
            return render_template('pages/signup.html')
        if User.query.filter((User.username==username)|(User.email==email)).first():
            flash('Username or email already registered.', 'error')
            return render_template('pages/signup.html')
        user = User(username=username, email=email, full_name=full_name,
                    password_hash=generate_password_hash(password), role='caregiver')
        db.session.add(user); db.session.commit()
        flash('Account created! Please log in.', 'success')
        return redirect(url_for('auth.login'))
    return render_template('pages/signup.html')

@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))
