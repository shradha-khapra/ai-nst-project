<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AdaIN Style Transfer</title>
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
<style>
    body {
        font-family: 'Inter', sans-serif;
        background-color: #0f172a;
        color: #e2e8f0;
        background-image: 
            linear-gradient(rgba(15, 23, 42, 0.9), rgba(15, 23, 42, 0.95)),
            url('https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?q=80&w=1974&auto=format&fit=crop');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        min-height: 100vh;
    }
    .header-section {
        padding: 5rem 0 3rem;
        text-align: center;
        position: relative;
    }
    .header-section h1 {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 3.5rem;
        background: linear-gradient(to right, #818cf8, #c084fc, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(192, 132, 252, 0.3);
        margin-bottom: 1rem;
    }
    .header-section .lead {
        font-size: 1.25rem;
        color: #94a3b8;
        max-width: 600px;
        margin: 0 auto;
    }
    .card {
        background: rgba(30, 41, 59, 0.6);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .card:hover {
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border-color: rgba(129, 140, 248, 0.3);
    }
    .card-header {
        background: rgba(255, 255, 255, 0.05);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        color: #f8fafc;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
        text-transform: uppercase;
        font-size: 0.9rem;
    }
    .preview-image {
        width: 100%;
        height: 300px;
        object-fit: contain;
        background-color: #0f172a;
        border: 2px dashed #334155;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #64748b;
        font-size: 1.2rem;
        overflow: hidden;
    }
    .preview-image img {
        width: 100%;
        height: 100%;
        object-fit: contain;
    }
    .btn-primary {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        border: none;
        padding: 15px 50px;
        font-size: 1.2rem;
        font-family: 'Orbitron', sans-serif;
        border-radius: 50px;
        transition: all 0.3s;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 10px 20px -10px rgba(168, 85, 247, 0.5);
    }
    .btn-primary:hover {
        background: linear-gradient(135deg, #4f46e5 0%, #9333ea 100%);
        box-shadow: 0 20px 30px -10px rgba(168, 85, 247, 0.6);
    }
    .btn-glow {
        animation: glowing 2s infinite;
    }
    @keyframes glowing {
        0% { box-shadow: 0 0 5px #a855f7; }
        50% { box-shadow: 0 0 20px #a855f7, 0 0 10px #6366f1; }
        100% { box-shadow: 0 0 5px #a855f7; }
    }
    /* Custom Range Slider */
    .range-wrap {
        position: relative;
        margin: 2rem auto;
        width: 100%;
        max-width: 90%;
    }
    
    #alphaRange {
        -webkit-appearance: none;
        width: 100%;
        height: 12px;
        background: rgba(30, 41, 59, 0.8);
        border-radius: 15px;
        outline: none;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: inset 0 2px 5px rgba(0,0,0,0.5);
        transition: background 0.1s;
    }
    
    #alphaRange::-webkit-slider-runnable-track {
        width: 100%;
        height: 100%;
        cursor: pointer;
        background: transparent;
    }
    
    #alphaRange::-webkit-slider-thumb {
        -webkit-appearance: none;
        height: 24px;
        width: 24px;
        border-radius: 50%;
        background: #fff;
        cursor: pointer;
        margin-top: -6px;
        box-shadow: 0 0 10px #c084fc, 0 0 20px #818cf8;
        border: 3px solid #1e293b;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    #alphaRange::-webkit-slider-thumb:hover {
        box-shadow: 0 0 15px #c084fc, 0 0 30px #818cf8;
    }

    .range-value {
        position: absolute;
        top: -40px;
        left: 50%;
        transform: translateX(-50%);
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 15px;
        font-family: 'Orbitron', sans-serif;
        font-size: 0.85rem;
        box-shadow: 0 5px 15px rgba(168, 85, 247, 0.4);
        opacity: 0;
        transition: opacity 0.3s;
        pointer-events: none;
    }
    .range-value::after {
        content: '';
        position: absolute;
        bottom: -5px;
        left: 50%;
        margin-left: -5px;
        border-width: 5px 5px 0;
        border-style: solid;
        border-color: #a855f7 transparent transparent transparent;
    }
    .range-wrap:hover .range-value {
        opacity: 1;
    }
    .form-control {
        background-color: #1e293b;
        border: 1px solid #334155;
        color: #f8fafc;
    }
    .form-control:focus {
        background-color: #1e293b;
        border-color: #818cf8;
        color: #f8fafc;
        box-shadow: 0 0 0 0.25rem rgba(129, 140, 248, 0.25);
    }
    .form-control::file-selector-button {
        background-color: #334155;
        color: #f8fafc;
        border: none;
        padding: 0.375rem 0.75rem;
        margin-right: 0.75rem;
        border-radius: 0.25rem;
        transition: background-color 0.2s;
    }
    .form-control::file-selector-button:hover {
        background-color: #475569;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #0f172a; 
    }
    ::-webkit-scrollbar-thumb {
        background: #334155; 
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #818cf8; 
    }

    /* Ambient Orbs */
    .orb {
        position: fixed;
        border-radius: 50%;
        filter: blur(100px);
        z-index: 0;
        opacity: 0.4;
        animation: floatOrb 15s infinite ease-in-out alternate;
        pointer-events: none;
    }
    .orb-1 { top: -10%; left: -10%; width: 600px; height: 600px; background: radial-gradient(circle, #4f46e5, transparent 70%); }
    .orb-2 { bottom: -10%; right: -10%; width: 500px; height: 500px; background: radial-gradient(circle, #c084fc, transparent 70%); animation-delay: -5s; }
    .orb-3 { top: 40%; left: 40%; width: 300px; height: 300px; background: radial-gradient(circle, #f472b6, transparent 70%); animation-delay: -10s; opacity: 0.2; }

    @keyframes floatOrb {
        0% { transform: translate(0, 0) scale(1); }
        100% { transform: translate(50px, 50px) scale(1.1); }
    }

    /* Scanline Effect */
    .scanlines {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(to bottom, rgba(255,255,255,0), rgba(255,255,255,0) 50%, rgba(0,0,0,0.2) 50%, rgba(0,0,0,0.2));
        background-size: 100% 4px;
        z-index: 9998;
        pointer-events: none;
        opacity: 0.6;
    }    

    /* Background Canvas */
    #neuralCanvas {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        z-index: 0;
        pointer-events: none;
    }
    .header-section, .container, .footer-section {
        position: relative;
        z-index: 1;
    }    

    /* Animations */
    .fade-in-up {
        animation: fadeInUp 0.8s ease-out forwards;
        opacity: 0;
        transform: translateY(30px);
    }
    
    @keyframes fadeInUp {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Loader */
    .loader-overlay {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(15, 23, 42, 0.9);
        backdrop-filter: blur(5px);
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        opacity: 0;
        pointer-events: none;
        transition: opacity 0.3s ease;
    }
    .loader-overlay.active {
        opacity: 1;
        pointer-events: all;
    }
    .cyber-spinner {
        width: 80px;
        height: 80px;
        border: 4px solid transparent;
        border-top: 4px solid #818cf8;
        border-right: 4px solid #c084fc;
        border-radius: 50%;
        animation: spin 1s linear infinite;
        box-shadow: 0 0 15px rgba(129, 140, 248, 0.5);
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    /* Navbar Styles */
    .navbar {
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }
    .navbar-brand {
        font-family: 'Orbitron', sans-serif;
        font-weight: 700;
        font-size: 1.5rem;
        background: linear-gradient(to right, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .nav-link {
        color: #e2e8f0 !important;
        font-weight: 500;
        margin-left: 1rem;
        transition: color 0.3s;
        position: relative;
    }
    .nav-link:hover {
        color: #818cf8 !important;
    }
    .nav-link::after {
        content: '';
        position: absolute;
        width: 0;
        height: 2px;
        bottom: 0;
        left: 0;
        background-color: #818cf8;
        transition: width 0.3s;
    }
    .nav-link:hover::after {
        width: 100%;
    }

    /* Footer Styles */
    .footer-section {
        background: rgba(15, 23, 42, 0.95);
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        padding: 4rem 0 2rem;
        margin-top: 5rem;
        color: #94a3b8;
    }
    .footer-section h5 {
        color: #f8fafc;
        font-family: 'Orbitron', sans-serif;
        margin-bottom: 1.5rem;
        font-size: 1.1rem;
    }
    .footer-link {
        color: #94a3b8;
        text-decoration: none;
        transition: color 0.3s;
        display: block;
        margin-bottom: 0.8rem;
    }
    .footer-link:hover {
        color: #818cf8;
    }
    .social-icon {
        width: 40px;
        height: 40px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: rgba(255,255,255,0.05);
        border-radius: 50%;
        color: #f8fafc;
        margin-right: 10px;
        transition: all 0.3s;
        text-decoration: none;
    }
    .social-icon:hover {
        background: #818cf8;
        color: white;
    }
    .examples-section {
        margin-top: 4rem;
    }
    .examples-section .example-image {
        width: 100%;
        height: 220px;
        object-fit: contain;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.12);
        background: rgba(15, 23, 42, 0.8);
    }
    .examples-section .example-role {
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #cbd5e1;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
    }
    .examples-section .tree-connector {
        color: #a5b4fc;
        font-size: 1.15rem;
        font-weight: 600;
        text-align: center;
        letter-spacing: 0.04em;
        margin: 0.2rem 0 0.4rem;
    }
    .examples-section .example-flow {
        display: flex;
        align-items: center;
        gap: 1.5rem;
        padding: 1.25rem;
        border-radius: 18px;
        background: rgba(15, 23, 42, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .examples-section .example-inputs {
        flex: 1;
        display: grid;
        gap: 1rem;
    }
    .examples-section .example-output {
        flex: 1;
    }
    .examples-section .example-arrow {
        color: #a5b4fc;
        font-size: 1.75rem;
        font-weight: 700;
        line-height: 1;
    }
    @media (max-width: 767.98px) {
        .examples-section .example-flow {
            flex-direction: column;
        }
        .examples-section .example-arrow {
            transform: rotate(90deg);
        }
    }
    .faq-section {
        margin-top: 4rem;
    }
    .faq-section .accordion-item {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 0.75rem;
        border-radius: 12px;
        overflow: hidden;
    }
    .faq-section .accordion-button {
        background: rgba(15, 23, 42, 0.75);
        color: #e2e8f0;
        font-weight: 600;
    }
    .faq-section .accordion-button:not(.collapsed) {
        background: rgba(129, 140, 248, 0.2);
        color: #f8fafc;
        box-shadow: none;
    }
    .faq-section .accordion-button:focus {
        box-shadow: 0 0 0 0.2rem rgba(129, 140, 248, 0.2);
    }
    .faq-section .accordion-body {
        color: #94a3b8;
    }
</style>
</head>
<body>

<canvas id="neuralCanvas"></canvas>
<div class="scanlines"></div>
<div class="orb orb-1"></div>
<div class="orb orb-2"></div>
<div class="orb orb-3"></div>

<nav class="navbar navbar-expand-lg fixed-top">
    <div class="container">
        <a class="navbar-brand" href="#"><i class="fas fa-brain me-2"></i>NeuralArt</a>
        <button class="navbar-toggler navbar-dark" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
            <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarNav">
            <ul class="navbar-nav ms-auto">
                <li class="nav-item"><a class="nav-link" href="#">Home</a></li>
                <li class="nav-item"><a class="nav-link" href="#">Gallery</a></li>
                <li class="nav-item"><a class="nav-link" href="#">Features</a></li>
                <li class="nav-item"><a class="nav-link" href="#">Pricing</a></li>
                <li class="nav-item"><a class="nav-link" href="#">Contact</a></li>
            </ul>
            <a href="#" class="btn btn-sm btn-outline-light ms-3 rounded-pill px-4">Sign In</a>
        </div>
    </div>
</nav>

<div class="header-section">
    <div class="container">
        <h1 class="display-4 fw-bold">STYLEFORGE AI</h1>
        <p class="lead" style="border-right: 2px solid rgba(255,255,255,0.75); white-space: nowrap; overflow: hidden; width: 0; animation: typing 3.5s steps(40, end) forwards, blink-caret .75s step-end infinite; margin: 0 auto; display: inline-block; text-align: left;">Redefine Reality with AI-Powered Artistry</p>
    </div>
</div>

<div class="container mb-5 fade-in-up">
    <form method="post" enctype="multipart/form-data" id="uploadForm">
        {{ form.hidden_tag() }}
        
        <div class="row g-4">
            <!-- Content Image Section -->
            <div class="col-md-6">
                <div class="card h-100">
                    <div class="card-header text-center py-3">Content Source</div>
                    <div class="card-body text-center">
                        <div class="preview-image mb-3" id="contentPreview">
                            {% if content_image %}
                                <img src="{{ url_for('send_image', filename=content_image) }}" alt="Content Image">
                            {% else %}
                                <span>Select Content Image</span>
                            {% endif %}
                        </div>
                        {{ form.content(class="form-control", onchange="previewFile(this, 'contentPreview')") }}
                    </div>
                </div>
            </div>

            <!-- Style Image Section -->
            <div class="col-md-6">
                <div class="card h-100">
                    <div class="card-header text-center py-3">Style Reference</div>
                    <div class="card-body text-center">
                        <div class="preview-image mb-3" id="stylePreview">
                            {% if style_image %}
                                <img src="{{ url_for('send_image', filename=style_image) }}" alt="Style Image">
                            {% else %}
                                <span>Select Style Image</span>
                            {% endif %}
                        </div>
                        {{ form.style(class="form-control", onchange="previewFile(this, 'stylePreview')") }}
                    </div>
                </div>
            </div>
        </div>

        <!-- Controls Section -->
        <div class="row justify-content-center mt-4">
            <div class="col-md-8">
                <div class="range-wrap">
                    <label for="alpha" class="form-label fw-bold mb-3 d-block text-center" style="font-family: 'Orbitron', sans-serif; letter-spacing: 1px; color: #a855f7;">STYLE STRENGTH</label>
                    <div class="position-relative">
                        <div class="range-value" id="rangeValue">1.0</div>
                        {{ form.alpha(class="form-control-range", id="alphaRange", type="range", min="0", max="1", step="0.1") }}
                    </div>
                </div>
                <div class="text-center mt-5">
                    {{ form.submit(class="btn btn-primary shadow-lg btn-glow") }}
                </div>
            </div>
        </div>
    </form>

    {% if result_image %}
    <div class="row justify-content-center mt-5" id="resultSection">
        <div class="col-md-8">
            <div class="card border-success">
                <div class="card-header bg-success text-white fw-bold text-center py-3" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);">Stylized Result</div>
                <div class="card-body p-0">
                    <img src="{{ url_for('send_image', filename=result_image) }}" class="img-fluid w-100" alt="Stylized Image" style="border-radius: 0 0 20px 20px;">
                </div>
                <div class="card-footer text-center" style="background: transparent; border-top: none; padding-bottom: 20px;">
                    <a href="{{ url_for('send_image', filename=result_image) }}" download class="btn btn-outline-light mt-2">Download Result</a>
                </div>
            </div>
        </div>
    </div>
    {% endif %}

    {% if error %}
        <div class="alert alert-danger mt-4 text-center" role="alert" style="background: rgba(220, 38, 38, 0.2); border-color: #dc2626; color: #fecaca;">
            <strong>Error:</strong> {{ error }}
        </div>
    {% endif %}
</div>

<section class="container examples-section fade-in-up">
    <div class="card">
        <div class="card-header text-center py-3">Examples</div>
        <div class="card-body">
            <p class="text-center text-muted mb-4">Each example shows the content image, the style below it, and the final output beside them.</p>
            <div class="row g-4">
                <div class="col-12">
                    <div class="example-flow">
                        <div class="example-inputs">
                            <div>
                                <div class="example-role text-center">Content Image</div>
                                <img src="{{ url_for('send_example', filename='brad_pitt.jpg') }}" alt="Brad Pitt Content" class="example-image">
                            </div>
                            <div>
                                <div class="example-role text-center">Style Image</div>
                                <img src="{{ url_for('send_example', filename='sketch.png') }}" alt="Sketch Style" class="example-image">
                            </div>
                        </div>
                        <div class="example-arrow">-&gt;</div>
                        <div class="example-output">
                            <div class="example-role text-center">Output</div>
                            <img src="{{ url_for('send_example', filename='stylized_brad_pitt.jpg') }}" alt="Brad Pitt Sketch Result" class="example-image">
                        </div>
                    </div>
                </div>
                <div class="col-12">
                    <div class="example-flow">
                        <div class="example-inputs">
                            <div>
                                <div class="example-role text-center">Content Image</div>
                                <img src="{{ url_for('send_example', filename='brad_pitt.jpg') }}" alt="Brad Pitt Content" class="example-image">
                            </div>
                            <div>
                                <div class="example-role text-center">Style Image</div>
                                <img src="{{ url_for('send_example', filename='picasso_seated_nude_hr.jpg') }}" alt="Picasso Seated Nude Style" class="example-image">
                            </div>
                        </div>
                        <div class="example-arrow">-&gt;</div>
                        <div class="example-output">
                            <div class="example-role text-center">Output</div>
                            <img src="{{ url_for('send_example', filename='stylized_brad_pitt (1).jpg') }}" alt="Brad Pitt Picasso Result" class="example-image">
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</section>

<section class="container faq-section fade-in-up">
    <div class="card">
        <div class="card-header text-center py-3">FAQs</div>
        <div class="card-body">
            <div class="accordion" id="faqAccordion">
                <div class="accordion-item">
                    <h2 class="accordion-header" id="faqHeadingOne">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faqOne" aria-expanded="false" aria-controls="faqOne">
                            1. Is it a pretrained model?
                        </button>
                    </h2>
                    <div id="faqOne" class="accordion-collapse collapse" aria-labelledby="faqHeadingOne" data-bs-parent="#faqAccordion">
                        <div class="accordion-body">
                            No, we train a model ourselves.
                        </div>
                    </div>
                </div>
                <div class="accordion-item">
                    <h2 class="accordion-header" id="faqHeadingTwo">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faqTwo" aria-expanded="false" aria-controls="faqTwo">
                            2. Is it a free platform?
                        </button>
                    </h2>
                    <div id="faqTwo" class="accordion-collapse collapse" aria-labelledby="faqHeadingTwo" data-bs-parent="#faqAccordion">
                        <div class="accordion-body">
                            This demo is currently available as a free experience.
                        </div>
                    </div>
                </div>
                <div class="accordion-item">
                    <h2 class="accordion-header" id="faqHeadingThree">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faqThree" aria-expanded="false" aria-controls="faqThree">
                            3. Which styles of painting can be used?
                        </button>
                    </h2>
                    <div id="faqThree" class="accordion-collapse collapse" aria-labelledby="faqHeadingThree" data-bs-parent="#faqAccordion">
                        <div class="accordion-body">
                            You can use almost any painting style image (impressionist, abstract, cubist, watercolor, sketch, and more) as the style reference.
                        </div>
                    </div>
                </div>
                <div class="accordion-item">
                    <h2 class="accordion-header" id="faqHeadingFour">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faqFour" aria-expanded="false" aria-controls="faqFour">
                            4. Tech stack of the project?
                        </button>
                    </h2>
                    <div id="faqFour" class="accordion-collapse collapse" aria-labelledby="faqHeadingFour" data-bs-parent="#faqAccordion">
                        <div class="accordion-body">
                            Python, PyTorch, Flask.
                        </div>
                    </div>
                </div>
                <div class="accordion-item">
                    <h2 class="accordion-header" id="faqHeadingFive">
                        <button class="accordion-button collapsed" type="button" data-bs-toggle="collapse" data-bs-target="#faqFive" aria-expanded="false" aria-controls="faqFive">
                            5. Which dataset / how big data?
                        </button>
                    </h2>
                    <div id="faqFive" class="accordion-collapse collapse" aria-labelledby="faqHeadingFive" data-bs-parent="#faqAccordion">
                        <div class="accordion-body">
                            The model is trained on large-scale image datasets.
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</section>

<footer class="footer-section">
    <div class="container">
        <div class="row g-4">
            <div class="col-lg-4 col-md-6">
                <h5><i class="fas fa-brain me-2"></i>NeuralArt</h5>
                <p class="mb-4">"Every artist dips his brush in his own soul, and paints his own nature into his pictures."</p>
                <p class="small text-muted">Empowering creators with state-of-the-art AI tools to reimagine reality through the lens of artistic expression.</p>
            </div>
            <div class="col-lg-2 col-md-6">
                <h5>Quick Links</h5>
                <a href="#" class="footer-link">Home</a>
                <a href="#" class="footer-link">About Us</a>
                <a href="#" class="footer-link">Showcase</a>
                <a href="#" class="footer-link">API Docs</a>
            </div>
            <div class="col-lg-3 col-md-6">
                <h5>Contact Us</h5>
                <p class="mb-2"><i class="fas fa-map-marker-alt me-2"></i> 123 Innovation Dr, Tech City</p>
                <p class="mb-2"><i class="fas fa-envelope me-2"></i> hello@neuralart.ai</p>
                <p class="mb-2"><i class="fas fa-phone me-2"></i> +1 (555) 123-4567</p>
            </div>
            <div class="col-lg-3 col-md-6">
                <h5>Follow Us</h5>
                <div class="d-flex mb-3">
                    <a href="#" class="social-icon"><i class="fab fa-twitter"></i></a>
                    <a href="#" class="social-icon"><i class="fab fa-facebook-f"></i></a>
                    <a href="#" class="social-icon"><i class="fab fa-instagram"></i></a>
                    <a href="#" class="social-icon"><i class="fab fa-github"></i></a>
                </div>
                <p class="small text-muted">Subscribe to our newsletter for the latest updates.</p>
            </div>
        </div>
        <hr class="my-4" style="border-color: rgba(255,255,255,0.1);">
        <div class="text-center text-muted small">
            &copy; 2025 NeuralArt Inc. All rights reserved. | Privacy Policy | Terms of Service
        </div>
    </div>
</footer>

<!-- Loading Overlay -->
<div class="loader-overlay" id="loader">
    <div class="cyber-spinner mb-4"></div>
    <h3 class="text-white" style="font-family: 'Orbitron', sans-serif; letter-spacing: 2px;">PROCESSING</h3>
    <p class="text-muted">Neural Network is dreaming...</p>
</div>

<style>
    /* Typing Animation Keyframes */
    @keyframes typing {
        from { width: 0 }
        to { width: 100% }
    }
    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: rgba(255,255,255,0.75); }
    }
</style>

<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
    function previewFile(input, previewId) {
        const preview = document.getElementById(previewId);
        const file = input.files[0];
        const reader = new FileReader();

        reader.addEventListener("load", function () {
            // Convert image file to base64 string
            preview.innerHTML = `<img src="${reader.result}" alt="Image Preview">`;
        }, false);

        if (file) {
            reader.readAsDataURL(file);
        }
    }

    // Restore previews if page reloads with data (optional enhancement)
    window.onload = function() {
        if(document.getElementById('resultSection')) {
            document.getElementById('resultSection').scrollIntoView({behavior: "smooth"});
        }
        // Neural Network Background Animation
        const canvas = document.getElementById('neuralCanvas');
        const ctx = canvas.getContext('2d');
        let width, height;
        let particles = [];
        let mouse = { x: null, y: null, radius: 150 };


        function resize() {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
        }

        class Particle {
            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.vx = (Math.random() - 0.5) * 0.5;
                this.vy = (Math.random() - 0.5) * 0.5;
                this.size = Math.random() * 3 + 1.5;
                // Assign one of 10 colors to group nodes
                this.hue = Math.floor(Math.random() * 10) * 36;
                this.density = (Math.random() * 30) + 1;
            }
            update() {
                this.x += this.vx;
                this.y += this.vy;

                // Mouse interaction
                if (mouse.x != null) {
                    let dx = mouse.x - this.x;
                    let dy = mouse.y - this.y;
                    let distance = Math.sqrt(dx * dx + dy * dy);
                    if (distance < mouse.radius) {
                        const forceDirectionX = dx / distance;
                        const forceDirectionY = dy / distance;
                        const force = (mouse.radius - distance) / mouse.radius;
                        const directionX = forceDirectionX * force * this.density;
                        const directionY = forceDirectionY * force * this.density;
                        this.x -= directionX;
                        this.y -= directionY;
                    }
                }

                if (this.x < 0 || this.x > width) this.vx *= -1;
                if (this.y < 0 || this.y > height) this.vy *= -1;
                this.hue = (this.hue + 0.2) % 360;
            }
            draw() {
                ctx.fillStyle = `hsla(${this.hue}, 70%, 60%, 0.3)`;
                ctx.beginPath();
                ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        function initParticles() {
            particles = [];
            const numParticles = Math.floor(window.innerWidth / 15);
            for (let i = 0; i < numParticles; i++) particles.push(new Particle());
        }

        function animate() {
            ctx.clearRect(0, 0, width, height);
            particles.forEach((p, i) => {
                p.update();
                p.draw();
                for (let j = i + 1; j < particles.length; j++) {
                    const p2 = particles[j];
                    const d = Math.hypot(p.x - p2.x, p.y - p2.y);
                    if (d < 100) {
                        ctx.strokeStyle = `hsla(${p.hue}, 70%, 60%, ${0.15 * (1 - d/100)})`;
                        ctx.lineWidth = 1;
                        ctx.beginPath();
                        ctx.moveTo(p.x, p.y);
                        ctx.lineTo(p2.x, p2.y);
                        ctx.stroke();
                    }
                }
            });
            requestAnimationFrame(animate);
        }

        window.addEventListener('resize', () => { resize(); initParticles(); });
        window.addEventListener('mousemove', (event) => {
            mouse.x = event.clientX;
            mouse.y = event.clientY;
        });        
        resize();
        initParticles();
        animate();
        // Range Slider Logic
        const range = document.getElementById('alphaRange');
        const rangeValue = document.getElementById('rangeValue');

        function updateRange() {
            const val = range.value;
            const min = range.min || 0;
            const max = range.max || 1;
            const percentage = ((val - min) * 100) / (max - min);
            
            rangeValue.textContent = val;
            // Adjust position of the bubble to follow the thumb
            const thumbWidth = 24;
            const offset = thumbWidth/2 - (percentage * thumbWidth / 100);
            rangeValue.style.left = `calc(${percentage}% + ${offset}px)`;
            
            // Update gradient background
            range.style.background = `linear-gradient(to right, #818cf8 0%, #c084fc ${percentage}%, rgba(30, 41, 59, 0.8) ${percentage}%, rgba(30, 41, 59, 0.8) 100%)`;
        }

        if(range) {
            range.addEventListener('input', updateRange);
            updateRange(); // Init
        }


    }
</script>
</body>
</html>
 
