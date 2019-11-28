const loading = document.getElementById('loading'),
	  loadingVisibleClass = 'loading--visible',
	  loadingHiddenClass = 'loading--hidden',
	  settingsmodal = document.getElementById('settingsmodal'),
	  modalHiddenClass = 'modal__container--hidden',
	  modalDisplayNone = 'modal__container--displaynone',
	  titleBar = document.getElementById('titlebar'),
	  titleBarActionsVisibleClass = 'titlebar--actionsvisible',
	  titleBarVisibleClass = 'titlebar--visible',
	  titlebaractionsButtonsDisplay = 'titlebar__actions--buttonsdisplay',
	  titlebaractionsButtonsShow = 'titlebar__actions--showbuttons',
	  stepper = document.getElementById('stepper');

const showLoading = () => {
	loading.classList.remove(loadingHiddenClass);
	setTimeout(() => {
		loading.classList.add(loadingVisibleClass);
	},100);
}

const hideLoading = () => {
	loading.classList.remove(loadingVisibleClass);
	setTimeout(() => {
		loading.classList.add(loadingHiddenClass);
	},500);
}

const showSettingsModal = () => {
	settingsmodal.classList.remove(modalDisplayNone);
	setTimeout(() => {
		settingsmodal.classList.remove(modalHiddenClass);
	},10);
}

const hideSettingsModal = () => {
	settingsmodal.classList.add(modalHiddenClass);
	setTimeout(() => {
		settingsmodal.classList.add(modalDisplayNone);
	},500);
}

const showTitleBar = () => {
	titleBar.classList.add(titleBarVisibleClass);
}

const hideTitleBar = () => {
	titleBar.classList.remove(titleBarVisibleClass);
}

const setStepper = (step) => {
	stepper.classList.remove('stepper--1','stepper--2','stepper--3');
	stepper.classList.add('stepper--' + step);
}

const switchSettingsModalToEdit = () => {
	settingsmodal.classList.add('modal--modify');
}

const switchForm = (formNumber) => {
	if (formNumber == 1) {
		document.getElementById('form--2').classList.add('form--waiting');
		document.getElementById('form--1').classList.remove('form--complete');
	}
	if (formNumber == 2) {
		document.getElementById('form--2').classList.remove('form--waiting');
		document.getElementById('form--1').classList.add('form--complete');
	}
}


// ALL CONTENT BELOW ARE FORCED EVENTS AS EXAMPLES

const testidtrigger = document.getElementById('testbuttonclick'),
	  titlebaricon = document.getElementById('actiontoggle'),
	  titlebarActions = document.getElementById('titlebar__actions'),
	  editbutton = document.getElementById('editbutton'),
	  backbutton = document.getElementById('backbutton');

// Dummy function to show loading state
testidtrigger.onclick = () => {
	if (stepper.classList.contains('stepper--2')) {
		showLoading();
		setTimeout(() => {
			hideLoading();
			hideSettingsModal();
			showTitleBar();
		}, 1500);
	} else {
		switchForm(2);
		setStepper(2);
	}
}

actiontoggle.onclick = () => {
	if (titlebar.classList.contains(titleBarActionsVisibleClass)) {
		titlebarActions.classList.remove(titlebaractionsButtonsShow);
		setTimeout(() => {
			titlebar.classList.remove(titleBarActionsVisibleClass);
			titlebarActions.classList.remove(titlebaractionsButtonsDisplay);
		}, 500);
	} else {
		titlebar.classList.add(titleBarActionsVisibleClass);
		titlebarActions.classList.add(titlebaractionsButtonsDisplay);
		setTimeout(() => {
			titlebarActions.classList.add(titlebaractionsButtonsShow);
		}, 10);
	}
}

editbutton.onclick = () => {
	switchSettingsModalToEdit();
	showSettingsModal();
}

backbutton.onclick = () => {
	setStepper(1);
	switchForm(1);
}



