function previewImage() {
	const input = document.getElementById('file');
	const output = document.getElementById('outputImage');
	output.innerHTML = '';

	if (input.files && input.files[0]) {
		const reader = new FileReader();
		reader.onload = function(e) {
			const img = new Image();
			img.src = e.target.result;
			output.appendChild(img);
		};
		reader.readAsDataURL(input.files[0]);
	}
}