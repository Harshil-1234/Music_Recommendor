(function ($) {
    "use strict";

    // Spinner
    var spinner = function () {
        setTimeout(function () {
            if ($('#spinner').length > 0) {
                $('#spinner').removeClass('show');
            }
        }, 1);
    };
    spinner();


    // Initiate the wowjs
    new WOW().init();


    // Sticky Navbar
    $(window).scroll(function () {
        if ($(this).scrollTop() > 300) {
            $('.sticky-top').addClass('shadow-sm').css('top', '0px');
        } else {
            $('.sticky-top').removeClass('shadow-sm').css('top', '-150px');
        }
    });

    // Modal Video
    var $videoSrc;
    $('.btn-play').click(function () {
        $videoSrc = $(this).data("src");
    });
    console.log($videoSrc);
    $('#videoModal').on('shown.bs.modal', function (e) {
        $("#video").attr('src', $videoSrc + "?autoplay=1&amp;modestbranding=1&amp;showinfo=0");
    })
    $('#videoModal').on('hide.bs.modal', function (e) {
        $("#video").attr('src', $videoSrc);
    })


    //Spotify
window.onSpotifyIframeApiReady = (IFrameAPI) => {
    const element = document.getElementById('embed-iframe');
    if (element) {
        const options = {
            width: '100%',
            height: '100%',
            uri: '{{ playlist_uri }}'
        };
        const callback = (EmbedController) => { };
        IFrameAPI.createController(element, options, callback);
    } else {
        console.error("Element with id 'embed-iframe' not found");
    }
};


    // Webcam
    const video = document.getElementById('webcam');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('capture-btn');
    const loadingSpinner = document.getElementById('loading-spinner');

    const constraints = {
        video: true
    };

    async function initWebcam() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
            video.classList.remove('d-none');
            loadingSpinner.style.display = 'none';
        } catch (error) {
            console.error('Error accessing webcam: ', error);
            alert('Error accessing webcam.');
        }
    }

    captureBtn.addEventListener('click', () => {
        const context = canvas.getContext('2d');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const dataURL = canvas.toDataURL('image/png');

        fetch('/try', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: dataURL }),
        })
            .then(response => response.json())
            .then(data => {
                if (data.redirect) {
                    const url = new URL(window.location.origin + data.redirect);
                    url.searchParams.append('emotion', data.emotion);
                    window.location.href = url.href;
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
    });

    window.addEventListener('load', () => {
        loadingSpinner.style.display = 'block';
        initWebcam();
    });

})(jQuery);

