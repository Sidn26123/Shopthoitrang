package ptithcm.entity;

import java.util.Date;
import java.util.List;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.FetchType;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;
import javax.persistence.JoinColumn;
import javax.persistence.ManyToOne;
import javax.persistence.OneToMany;
import javax.persistence.Table;
import javax.persistence.Temporal;
import javax.persistence.TemporalType;

import org.springframework.format.annotation.DateTimeFormat;

@Entity
@Table(name = "DONHANG")
public class DonHangEntity{
	@Id
	@Column(name="MADH")
	@GeneratedValue
	private int maDh;
	@Temporal(TemporalType.DATE)
	@DateTimeFormat(pattern = "yyyy-mm-dd")
	@Column(name="NGAYTAO")
	private Date ngayTao;
	
	@Column(name="HOTEN")
	private String hoTen;
	@Column(name="SDT")
	private String sdt;
	@Column(name="DIACHI")
	private String diaChi;
	@Column(name = "TONGTIEN")
	private int tongTien;
	@Column(name = "TRANGTHAI")
	private int trangThai;
	@ManyToOne
	@JoinColumn(name="MAND")
	private NguoiDungEntity nguoiDung;
	@OneToMany(mappedBy = "donHang",fetch = FetchType.EAGER)
	private List<CTDonHangEntity> ctDonHangList;
	public int getMaDh() {
		return maDh;
	}
	public void setMaDh(int maDh) {
		this.maDh = maDh;
	}
	public Date getNgayTao() {
		return ngayTao;
	}
	public void setNgayTao(Date ngayTao) {
		this.ngayTao = ngayTao;
	}
	public String getHoTen() {
		return hoTen;
	}
	public void setHoTen(String hoTen) {
		this.hoTen = hoTen;
	}
	public String getSdt() {
		return sdt;
	}
	public void setSdt(String sdt) {
		this.sdt = sdt;
	}
	public String getDiaChi() {
		return diaChi;
	}
	public void setDiaChi(String diaChi) {
		this.diaChi = diaChi;
	}
	public int getTongTien() {
		return tongTien;
	}
	public void setTongTien(int tongTien) {
		this.tongTien = tongTien;
	}
	public int getTrangThai() {
		return trangThai;
	}
	public void setTrangThai(int trangThai) {
		this.trangThai = trangThai;
	}
	public NguoiDungEntity getNguoiDung() {
		return nguoiDung;
	}
	public void setNguoiDung(NguoiDungEntity nguoiDung) {
		this.nguoiDung = nguoiDung;
	}
	public List<CTDonHangEntity> getCtDonHangList() {
		return ctDonHangList;
	}
	public void setCtDonHangList(List<CTDonHangEntity> ctDonHangList) {
		this.ctDonHangList = ctDonHangList;
	}
	
	
}